"""Exact test/evaluation/heatmap/feature-map functions extracted from the original notebook."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F_torch
from skimage.measure import label, regionprops
from IPython.display import display
from .model import UnetBlocks
from .losses_metrics import DiceScore, nIoU_Score, pd_score

# >>> ORIGINAL NOTEBOOK CELL 70 START (UNMODIFIED)
def results(image_tensors, mask_tensors, model_pth, device):

    model = UnetBlocks(2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device), strict=False)
    model.eval()

    nIoUs = []
    dice_scores = []
    pd_scores = []

    total_intersection = 0.0
    total_union = 0.0

    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0

    eps = 1e-8
    dice_calculator = DiceScore(eps=eps).to(device)

    with torch.no_grad():

        for image_pth, mask_pth in zip(image_tensors, mask_tensors):

            image = image_pth.unsqueeze(0).to(device)
            mask = mask_pth.unsqueeze(0).to(device).float()

            mean, log_var, bot_head, bot_ddpm, head2, head3, head4 = model(image)

            niou = nIoU_Score(mean, mask)
            nIoUs.append(float(niou))
            pd_scores.append(pd_score(mean, mask))

            dice_score = dice_calculator(mean, mask)
            dice_scores.append(float(dice_score))

            pred = (torch.sigmoid(mean) > 0.5).float()

            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum() - intersection

            total_intersection += intersection.item()
            total_union += union.item()

            TP = (pred * mask).sum()
            FP = (pred * (1 - mask)).sum()
            FN = ((1 - pred) * mask).sum()

            total_TP += TP.item()
            total_FP += FP.item()
            total_FN += FN.item()

    IoU = total_intersection / (total_union + eps)
    precision = total_TP / (total_TP + total_FP + eps)
    recall = total_TP / (total_TP + total_FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    results = pd.DataFrame([[IoU,np.mean(nIoUs),precision,recall,f1,np.mean(dice_scores),np.mean(pd_scores)
        ]],columns=["IoU","nIoU","Precision","Recall","F1","Dice Score","Pd"],index=["Test Set"])

    print("\nEvaluation Results")
    display(results.round(5))
# <<< ORIGINAL NOTEBOOK CELL 70 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 74 START (UNMODIFIED)
def draw_bboxes(ax, binary_mask, color="red"):
    labels = label(binary_mask)
    regions = regionprops(labels)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox

        rect = patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor=color,
            linewidth=1
        )

        ax.add_patch(rect)
# <<< ORIGINAL NOTEBOOK CELL 74 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 75 START (UNMODIFIED)
def heatmaps(image_tensors, mask_tensors, model_pth, device):

    model = UnetBlocks(2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device), strict=False)
    model.eval()

    with torch.no_grad():

        for image_pth, mask_pth in zip(image_tensors, mask_tensors):

            image = image_pth.unsqueeze(0).to(device)
            mask = mask_pth.unsqueeze(0).to(device).float()

            mean, log_var, bot_head, bot_ddpm, head2, head3, head4 = model(image)

            img = image_pth.cpu().permute(1, 2, 0)

            pred_show = torch.sigmoid(mean.squeeze(0)).cpu()
            pred_show = (pred_show > 0.5).float().permute(1, 2, 0)

            bot_head = F_torch.interpolate(bot_head, size=img.shape[:2], mode="bilinear", align_corners=False)
            bot_heatmap = (bot_head[0, 0].float().detach().cpu() - bot_head[0, 0].float().detach().cpu().min()) / (bot_head[0, 0].float().detach().cpu().max() - bot_head[0, 0].float().detach().cpu().min() + 1e-8)

            bot_ddpm = F_torch.interpolate(bot_ddpm, size=img.shape[:2], mode="bilinear", align_corners=False)
            bot_heatmap_ddpm = (bot_ddpm[0, 0].float().detach().cpu() - bot_ddpm[0, 0].float().detach().cpu().min()) / (bot_ddpm[0, 0].float().detach().cpu().max() - bot_ddpm[0, 0].float().detach().cpu().min() + 1e-8)

            mask_show = mask_pth.cpu().permute(1, 2, 0)

            pred_np = pred_show.squeeze().numpy().astype("uint8")
            mask_np = mask_show.squeeze().numpy().astype("uint8")

            plt.figure(figsize=(20, 6))

            ax1 = plt.subplot(151)
            ax1.imshow(img, cmap="gray")
            ax1.set_title("Original Image")
            ax1.axis("off")

            ax2 = plt.subplot(152)
            ax2.imshow(pred_show, cmap="gray")
            ax2.set_title("Prediction + BBox")
            draw_bboxes(ax2, pred_np, color="red")
            ax2.axis("off")

            ax3 = plt.subplot(153)
            ax3.imshow(mask_show, cmap="gray")
            ax3.set_title("Ground Truth + BBox")
            draw_bboxes(ax3, mask_np, color="lime")
            ax3.axis("off")

            ax4 = plt.subplot(154)
            heatmap_plot = ax4.imshow(bot_heatmap.numpy(),cmap="jet",vmin=0,vmax=1)
            ax4.set_title("Bottleneck Head Heatmap")
            ax4.axis("off")

            plt.colorbar(heatmap_plot,ax=ax4,fraction=0.046,pad=0.04)

            ax5 = plt.subplot(155)
            heatmap_plot = ax5.imshow(bot_heatmap_ddpm.numpy(),cmap="jet",vmin=0,vmax=1)
            ax5.set_title("Bottleneck DDPM Heatmap")
            ax5.axis("off")

            plt.colorbar(heatmap_plot,ax=ax5,fraction=0.046,pad=0.04)

            plt.tight_layout()
            plt.show()
# <<< ORIGINAL NOTEBOOK CELL 75 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 78 START (UNMODIFIED)
def save_feature(name,features,hooks):
  def hook(module, input, output):
      features.append(output.detach().cpu())
  return hook
# <<< ORIGINAL NOTEBOOK CELL 78 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 79 START (UNMODIFIED)
def feature_map_inference(image_tensors, mask_tensors, important_layers, model_pth, device):

    model = UnetBlocks(2).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device), strict=False)
    model.eval()

    features = []
    hooks = []

    for name, layer in model.named_modules():
        if name in important_layers:
            hooks.append(layer.register_forward_hook(save_feature(name,features,hooks)))

    image_tensors = image_tensors[0]
    x_input = image_tensors.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_input)

    for h in hooks:
        h.remove()

    img = image_tensors.cpu().detach().permute(1, 2, 0)
    mask = mask_tensors[0].cpu().detach().permute(1, 2, 0)

    total_plots = 2 + len(features)

    fig = plt.figure(figsize=(5 * total_plots, 5))


    ax = fig.add_subplot(1, total_plots, 1)
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title("Original")
    ax.axis("off")


    ax = fig.add_subplot(1, total_plots, 2)
    ax.imshow(mask.squeeze(), cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    for idx, (name, fmap) in enumerate(zip(important_layers, features), start=3):

        fmap = fmap.max(dim=1)[0][0]
        fmap = fmap.numpy()

        H, W = fmap.shape
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        ax = fig.add_subplot(1, total_plots, idx, projection='3d')
        ax.plot_surface(X, Y, fmap, cmap='jet')

        ax.set_title(f"{name}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()
    plt.show()
# <<< ORIGINAL NOTEBOOK CELL 79 END (UNMODIFIED)

