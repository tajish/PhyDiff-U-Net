"""Verbatim model class definitions from notebooks/Unet_Wavelet_SIRST(1).ipynb.

The code between the ORIGINAL NOTEBOOK CELL markers is UNMODIFIED.
Only the standalone-module imports and the Colab device global below were added.
The original cell uses a global ``device`` and assumes 256 x 256 input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# >>> ORIGINAL NOTEBOOK CELL 18 START (UNMODIFIED)
class PG_VEM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        self.channels = channels

        # Estimate transmittance map from [x, x_dark]
        self.t_estimator = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=1)
        )

        # Predict channel-level alpha using GAP
        hidden = max(channels // reduction, 4)
        self.alpha_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1)
        )

    def dark_channel(self, x):
        return -F_torch.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_dark = self.dark_channel(x)

        fused = torch.cat([x, x_dark], dim=1)

        tmap = torch.sigmoid(self.t_estimator(fused))

        alpha = self.alpha_predictor(x)

        # physics-informed enhancement weight
        W = 1 + torch.tanh(alpha) * 2.0 * (1 - tmap)

        # enhance features
        out = x * W

        return out
# <<< ORIGINAL NOTEBOOK CELL 18 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 20 START (UNMODIFIED)
class BottleneckDiffusionEnhancement(nn.Module):
    def __init__(self,channels,prior_channels,hidden_channels=32,noise_std=0.10,):
        super().__init__()

        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Conv2d(channels + prior_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False,),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False,),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels,channels,kernel_size=3,padding=1,),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, bottleneck, prior):

        if self.training and self.noise_std > 0:
            noisy_feature = (bottleneck + torch.randn_like(bottleneck) * self.noise_std)
        else:
            noisy_feature = bottleneck

        conditioned = torch.cat([noisy_feature, prior],dim=1,)

        feature = self.encoder(conditioned)
        residual = self.decoder(feature)

        return bottleneck + self.residual_scale * residual
# <<< ORIGINAL NOTEBOOK CELL 20 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 22 START (UNMODIFIED)
class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, dropout=0.3):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 1, 1)
        )
        self.var_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 1, 1)
        )
    def forward(self, x):
        mean = self.mean_head(x)
        log_var = torch.clamp(self.var_head(x), -10.0, 10.0)
        return mean, log_var
# <<< ORIGINAL NOTEBOOK CELL 22 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 24 START (UNMODIFIED)
class EdgeDetection(nn.Module):
    def __init__(self,channels):
        super().__init__()

        Kx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        Ky = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32)

        self.register_buffer("Kx", Kx.view(1, 1, 3, 3))
        self.register_buffer("Ky", Ky.view(1, 1, 3, 3))
        self.channels = channels
        self.conv_layer = nn.Conv2d(self.channels, 1, kernel_size=1,bias=False)

    def forward(self, x):

        B, C, H, W = x.shape

        Kx = self.Kx.repeat(C, 1, 1, 1)
        Ky = self.Ky.repeat(C, 1, 1, 1)

        Gx = F_torch.conv2d(x, Kx, padding=1, groups=C)
        Gy = F_torch.conv2d(x, Ky, padding=1, groups=C)

        G = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-6)

        patches = F_torch.unfold(G, kernel_size=3, padding=1)  # B, C*9, H*W
        patches = patches.view(B, C, 9, H, W)

        center = G.unsqueeze(2)  # B, C, 1, H, W
        diff = torch.abs(patches - center)

        # remove center pixel from 3x3 window
        mask = torch.ones(9, device=x.device, dtype=torch.bool)
        mask[4] = False
        diff = diff[:, :, mask, :, :]  # B, C, 8, H, W

        M = diff.mean(dim=2)
        S = torch.sqrt(((diff - M.unsqueeze(2)) ** 2).mean(dim=2) + 1e-6)

        G_mean = G.mean(dim=(2, 3), keepdim=True)
        S_mean = S.mean(dim=(2, 3), keepdim=True)

        condition = (G >= G_mean) & (S >= S_mean)

        D = torch.where(condition, G + S, torch.zeros_like(G))

        D = torch.sigmoid(D)

        x = x * D

        return x
# <<< ORIGINAL NOTEBOOK CELL 24 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 26 START (UNMODIFIED)
class SobelBoundaryAttention(nn.Module):
    def __init__(self,device):
        super().__init__()

        Kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32) / 2

        Ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32) / 2

        self.register_buffer("Kx", Kx.view(1, 1, 3, 3))
        self.register_buffer("Ky", Ky.view(1, 1, 3, 3))

    def forward(self, x):
        B, C, H, W = x.shape

        Kx = self.Kx.repeat(C, 1, 1, 1)
        Ky = self.Ky.repeat(C, 1, 1, 1)

        Mx = F_torch.conv2d(x, Kx, stride=1, padding=1, groups=C)
        My = F_torch.conv2d(x, Ky, stride=1, padding=1, groups=C)

        Mxy = torch.sqrt(Mx ** 2 + My ** 2 + 1e-6)

        attention = torch.sigmoid(Mxy)

        out = x * attention

        return out
# <<< ORIGINAL NOTEBOOK CELL 26 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 28 START (UNMODIFIED)
class DWT(nn.Module):
    def __init__(self, device):
        super().__init__()

        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2

        filters = torch.stack([ll, lh, hl, hh], dim=0)
        filters = filters.unsqueeze(1)

        self.register_buffer("filters", filters.to(device))

    def forward(self, x):
        B, C, H, W = x.shape

        weight = self.filters.repeat(C, 1, 1, 1)

        out = F_torch.conv2d(x,weight,stride=2,groups=C)

        return out
# <<< ORIGINAL NOTEBOOK CELL 28 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 30 START (UNMODIFIED)
class PhysicsPriorExtraction(nn.Module):
    def __init__(self,channel, psf_sigma=1.0, r_in=3, r_out=9, use_wavelet=True):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.register_buffer('psf_kernel', self._create_gaussian_kernel(psf_sigma))
        self.register_buffer('annulus_mask', self._create_annulus_mask(r_in, r_out))
        self.channel = channel
        if self.use_wavelet:
            self.dwt = DWT(device)
            self.wavelet_proj = nn.Conv2d(4, 1, kernel_size=1)
        else:
            self.high_pass = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.LeakyReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
            )
        self.prior_proj = nn.Conv2d(2, self.channel, kernel_size=1)

    def _create_gaussian_kernel(self, sigma, size=7):
        x = torch.linspace(-(size//2), size//2, size)
        y = torch.linspace(-(size//2), size//2, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

    def _create_annulus_mask(self, r_in, r_out):
        size = 2*r_out + 1
        x = torch.linspace(-r_out, r_out, size)
        y = torch.linspace(-r_out, r_out, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        mask = ((dist >= r_in) & (dist <= r_out)).float()
        return (mask / (mask.sum() + 1e-8)).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        psf_conv = F_torch.conv2d(x, self.psf_kernel, padding=self.psf_kernel.shape[-1]//2)
        local_mean = F_torch.conv2d(x, self.annulus_mask, padding=self.annulus_mask.shape[-1]//2)
        thermal = psf_conv - local_mean
        thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
        if self.use_wavelet:
            dwt_out = self.dwt(x)
            freq = self.wavelet_proj(dwt_out)
            freq = F_torch.interpolate(freq, size=x.shape[2:], mode='bilinear', align_corners=True)
        else:
            freq = self.high_pass(x)
        freq = (freq - freq.min()) / (freq.max() - freq.min() + 1e-8)
        prior = torch.cat([thermal, freq], dim=1)
        prior = self.prior_proj(prior)
        return prior
# <<< ORIGINAL NOTEBOOK CELL 30 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 32 START (UNMODIFIED)
class IWT(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        B, C, H, W = x.shape

        out_c = C // 4

        ll = x[:, 0:out_c, :, :]
        lh = x[:, out_c:2*out_c, :, :]
        hl = x[:, 2*out_c:3*out_c, :, :]
        hh = x[:, 3*out_c:4*out_c, :, :]

        h = torch.zeros(B, out_c, H * 2, W * 2, device=x.device)

        h[:, :, 0::2, 0::2] = ll - lh - hl + hh
        h[:, :, 0::2, 1::2] = ll - lh + hl - hh
        h[:, :, 1::2, 0::2] = ll + lh - hl - hh
        h[:, :, 1::2, 1::2] = ll + lh + hl + hh

        return h / 2
# <<< ORIGINAL NOTEBOOK CELL 32 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 34 START (UNMODIFIED)
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]

        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        l = self.gap(x)   # [B, C, 1, 1]

        z = self.fc(l)    # [B, C, 1, 1]

        importance = self.sigmoid(z) # [B, C, 1, 1]

        out = x * importance

        return out
# <<< ORIGINAL NOTEBOOK CELL 34 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 36 START (UNMODIFIED)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1,kernel_size=kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg_map, max_map], dim=1)
        attention = self.sigmoid(self.conv(attention))

        return x * attention
# <<< ORIGINAL NOTEBOOK CELL 36 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 38 START (UNMODIFIED)
class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.shift = nn.Parameter(torch.ones(emb_dim))
        self.scale = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.shift * normalized + self.scale
# <<< ORIGINAL NOTEBOOK CELL 38 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 40 START (UNMODIFIED)
class FeedForward(nn.Module):
  def __init__(self,channels):
    super().__init__()
    self.channels = channels
    self.layers = nn.Sequential(nn.Sequential(nn.Linear(self.channels, 4 * self.channels),
                                              nn.GELU(),
                                              nn.Sequential(nn.Linear(4 * self.channels,self.channels))
                                              ))
  def forward(self,x):
    return self.layers(x)
# <<< ORIGINAL NOTEBOOK CELL 40 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 42 START (UNMODIFIED)
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_patches, n_heads, dim_in, dim_out):
        super().__init__()

        assert dim_out % n_heads == 0

        self.n_patches = n_patches
        self.dim_out = dim_out
        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads

        self.q_proj = nn.Linear(dim_in, 1, bias=False)
        self.k_proj = nn.Linear(dim_in, dim_out, bias=False)
        self.v_proj = nn.Linear(dim_in, dim_out, bias=False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        q = self.q_proj(x)          # [B, N, 1]
        K = self.k_proj(x)          # [B, N, dim_out]
        V = self.v_proj(x)          # [B, N, dim_out]

        K = K.view(B, N, self.n_heads, self.head_dim)
        V = V.view(B, N, self.n_heads, self.head_dim)

        K = K.permute(0, 2, 1, 3)   # [B, heads, N, head_dim]
        V = V.permute(0, 2, 1, 3)   # [B, heads, N, head_dim]

        q = torch.softmax(q, dim=1) # [B, N, 1]
        q = q.unsqueeze(1)          # [B, 1, N, 1]

        context = torch.sum(K * q, dim=2, keepdim=True)  # [B, heads, 1, head_dim]

        context = torch.softmax(context / (self.head_dim ** 0.5),dim=-1)

        context = self.dropout(context)

        out = V * context           # [B, heads, N, head_dim]

        out = out.permute(0, 2, 1, 3)  # [B, N, heads, head_dim]
        out = out.contiguous().view(B, N, self.dim_out) # [B, N, dim_out]

        return out
# <<< ORIGINAL NOTEBOOK CELL 42 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 44 START (UNMODIFIED)
class TransformerBlock(nn.Module):
    def __init__(self,n_tokens,channels):
        super().__init__()

        self.dim_in = channels
        self.dim_out = channels
        self.n_tokens = n_tokens
        self.norm1 = nn.LayerNorm(self.dim_out)
        self.norm2 = nn.LayerNorm(self.dim_out)
        self.atten = MultiHeadAttentionLayer(self.n_tokens,1,self.dim_in,self.dim_out)
        self.ffn = FeedForward(self.dim_out)
        self.drop_short = nn.Dropout(0.1)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.atten(x)
        x = self.drop_short(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_short(x)
        x = x + shortcut
        return x
# <<< ORIGINAL NOTEBOOK CELL 44 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 46 START (UNMODIFIED)
class Transformer(nn.Module):
    def __init__(self,channels,Height,Width):
        super().__init__()

        self.channels = channels
        self.Height = Height
        self.Width = Width
        self.tokens = self.Height * self.Width


        self.linear_projection = nn.Sequential(
            nn.Linear(self.channels, 4 * self.channels),
            nn.GELU(),
            nn.Linear(4 * self.channels, self.channels)
        )


        self.positional_embedding = nn.Embedding(self.tokens, self.channels)

        self.dropout = nn.Dropout(0.1)

        self.pos_scale = nn.Parameter(torch.tensor(0.1))

        self.trf_block = nn.Sequential(*[TransformerBlock(self.tokens,self.channels) for l in range(1)])

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.flatten(2)       # (B, 256, H*W)
        x = x.transpose(1, 2)  # (B, H*W, 256)

        shortcut = x
        x = self.linear_projection(x)
        x = x + shortcut

        pos_ids = torch.arange(x.size(1), device=x.device)

        pos_embed = self.positional_embedding(pos_ids)

        x = x + self.pos_scale * pos_embed

        x = self.dropout(x)

        x = self.trf_block(x)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(B, C, H, W)

        return x
# <<< ORIGINAL NOTEBOOK CELL 46 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 48 START (UNMODIFIED)
class UnetBlocks(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3,padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3,padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.layer_7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),  # 64 + 32
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(inplace=True),
        )

        self.layer_8 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),  # 32 + 16
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),
        )

        self.layer_9 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1, bias=False),  # 16 + 8
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True),
        )


        self.DWT = DWT(device)
        self.IWT = IWT()

        self.transformer = Transformer(channels=256,Height=32,Width=32)
        self.ddpm = BottleneckDiffusionEnhancement(256,256)

        self.PPE1 = PhysicsPriorExtraction(1)
        self.PPE2 = PhysicsPriorExtraction(64)
        self.PPE3 = PhysicsPriorExtraction(128)
        self.PPE4 = PhysicsPriorExtraction(256)

        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()

        self.head2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4)
        self.head3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        self.head4 = nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1)

        self.SBA = SobelBoundaryAttention(device)

        self.UN = UncertaintyHead(16)

        #self.ac1 = nn.Conv2d(256, 1, kernel_size=1,bias=False)
        #self.ac2 = nn.Conv2d(256, 1, kernel_size=1,bias=False)

    def forward(self, x):

        x_1 = self.PPE1(x)               # [B, 1, 256, 256]
        x = torch.cat([x, x_1], dim=1)   # [B, 2, 256, 256]
        x1 = self.layer_1(x)             # [B, 16, 256, 256]
        p1 = self.DWT(x1)                # [B, 64, 128, 128]

        p1_prior = self.PPE2(p1)
        p1 = p1 + p1_prior
        x2 = self.layer_2(p1)        # [B, 32, 128, 128]
        p2 = self.DWT(x2)            # [B, 128, 64, 64]

        p2_prior = self.PPE3(p2)
        p2 = p2 + p2_prior
        x3 = self.layer_3(p2)        # [B, 64, 64, 64]
        p3 = self.DWT(x3)            # [B, 256, 32, 32]

        p3_prior = self.PPE4(p3)                                    #[B, 256, 32, 32]
        p3 = p3 + p3_prior                                          #[B, 256, 32, 32]
        prior_tensor = p3_prior                                     #[B, 256, 32, 32]
        bottleneck = self.bottleneck(p3)                            #[B, 256, 32, 32]

        #Transformer
        bottleneck = self.transformer(bottleneck)                   #[B, 256, 32, 32]
        bot_head = bottleneck.mean(dim=1,keepdim=True)              #[B, 1, 32, 32]

        bottleneck = self.ddpm(bottleneck, prior_tensor)            #[B,256,32,32]
        bot_ddpm = bottleneck.mean(dim=1,keepdim=True)              #[B, 1, 32, 32]

        x7 = self.IWT(bottleneck)      # [B, 64, 64, 64]
        x3 = self.SA2(x3)               # [B, 64, 64, 64]
        x7 = torch.cat([x3, x7], dim=1) # [B, 128, 64, 64]
        x7 = self.SBA(x7)             # [B, 128, 64, 64]
        x7 = self.layer_7(x7)         # [B, 64, 64, 64]
        head2 = self.head2(x7)        # [B, 1, 256, 256]

        x8 = self.IWT(x7)             # [B, 16, 128, 128]
        x2 = self.SA3(x2)             # [B, 32, 128, 128]
        x8 = torch.cat([x2, x8], dim=1) # [B, 48, 128, 128]
        x8 = self.SBA(x8)             # [B, 48, 128, 128]
        x8 = self.layer_8(x8)         # [B, 32, 128, 128]
        head3 = self.head3(x8)        # [B, 1, 256, 256]

        x9 = self.IWT(x8)               # [B, 8, 256, 256]
        x1 = self.SA4(x1)               # [B, 16, 256, 256]
        x9 = torch.cat([x1, x9], dim=1) # [B, 24, 256, 256]
        x9 = self.SBA(x9)               # [B, 24, 256, 256]
        x9 = self.layer_9(x9)           # [B, 16, 256, 256]
        head4 = self.head4(x9)          # [B, 1, 256, 256]

        mean, log_var = self.UN(x9)

        return mean, log_var, bot_head, bot_ddpm, head2, head3, head4
# <<< ORIGINAL NOTEBOOK CELL 48 END (UNMODIFIED)

