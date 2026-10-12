# Reproducibility and fidelity notes

## Provenance and limits

The training and experiment notebooks are included **unchanged**. Original model, loss, metric and plotting function bodies are extracted verbatim into the separate Python modules. Independent helper scripts were added, and must **not** be mistaken for original experimental code.

This bundle does **not** include raw infrared datasets, pretrained network checkpoint files (`my_checkpoint.pth`, `ACM.pt`, `ALCNet.pt`, `DNA-NET.pt`, `SWIN-UNET.pt`, `Ours.pt`), published per-sample prediction tensors, or the scripts/data which created the accuracy–FLOPs scatter plot. The eight supplied figure PNGs have been copied unchanged. Without missing inputs, there is no way to independently verify the published scores or recreate every plot.

## Source code vs. manuscript discrepancies — intentionally NOT corrected

- **FRE perturbation**: manuscript section 3.3 says `sigma_n` initialized to 0.2; original cell 20 defaults to **0.10**.
- **Heteroscedastic Monte Carlo**: manuscript section 3.7 states `K=10`; training cell 63 instantiates `HeteroscedasticBCELoss(num_samples=1)`.
- **Uncertainty regularizer**: manuscript equation (53) uses `mean(exp(s))`; notebook cell 62 uses **`log_var.mean()`**.
- **Loss description**: manuscript section 3.7 mentions focal+Dice segmentation loss and heteroscedastic BCE. Notebook cell 67 actually uses **`HeteroscedasticBCELoss` for the main loss** and **`EdgeLoss` for deep supervision** with weights `(0.2,0.3,0.4)/2`.
- **Datasets**: training notebook cell 9 uses a **90:5:5** split regardless of dataset. The manuscript table reports **90:5:5** for NUAA-SIRST and **80:10:10** for NUDT-SIRST and IRSTS-1K. The provided training notebook uses a `/content/sirst dataset/` path and `Misc_*.png` naming, not generic dataset adapters.
- **Input and positional embeddings**: original `UnetBlocks` hardcodes the Transformer positional embedding for a 32x32 bottleneck: use **256x256 grayscale input**.
- **3D experiment notebook**: `selected_files = ["XDU689.png", "XDU296.png"]`; its functions require five pretrained TorchScript models and the correct dataset directories. Its example cells use `device="cuda"`.
- **Training notebook import order**: the last imported `label` comes from `skimage.measure`, while the experiment notebook reimports `label` from `scipy.ndimage`; extracted Python modules preserve that import resolution explicitly.
- **No auto-run guarantee for notebooks**: the original notebooks include Colab-only imports (`google.colab.patches`), absolute `/content` paths, and results outputs; these are deliberately not edited.

These differences and constraints are documented rather than silently modified, as requested. Use the original notebooks and the original training datasets/checkpoints for a strict reproduction claim.

## Security/data hygiene

Do not commit private datasets, credentials, very large pretrained models, or experiment access tokens. `.gitignore` excludes common checkpoint and ZIP types. External model weights should be reviewed before loading, especially TorchScript files from untrusted sources.

- **TorchScript export**: `torch.jit.script` fails on the unmodified PPE class because the inactive optional `high_pass` branch is conditionally created. The separate export helper uses `torch.jit.trace` at 256×256, leaving the model untouched.
- **Notebook-format warning**: the original training notebook contains nonstandard `metadata` on at least one `stream` output, so strict `nbformat.validate()` rejects it; we preserve the original bytes and do not strip outputs or metadata.
