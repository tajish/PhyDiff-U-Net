# PhyDiff U-Net: Original Notebook Implementation

**Paper:** *PhyDiff U-Net: A Physics-Informed Diffusion-Inspired Transformer for Infrared Small-Target Segmentation*

This repository is assembled from the **actual Colab notebooks and eight figure images supplied by the paper's author**. The architecture and experiment functions below have been extracted *verbatim* from the originals; the notebooks themselves are **byte-for-byte unchanged**. No architecture was redesigned and no method was silently corrected.

> **Important:** The provided notebooks are the source of truth. This is a research-code release, not a verified reproduction of the paper's metrics. No training data, model checkpoints, or baseline implementations were supplied.

## Architecture

![PhyDiff U-Net architecture](assets/figures/Figure%201.png)

The original notebook's `UnetBlocks(2)` predicts **7 outputs**:
`mean, log_var, bot_head, bot_ddpm, head2, head3, head4`.

**Implemented in the original notebook:** physics-prior extraction (thermal PSF-annulus contrast + Haar-frequency features); DWT downsampling; separable-attention Transformer bottleneck; noise-perturbation-guided feature refinement (FRE); IWT upsampling; spatial skip attention; Sobel boundary attention; deep-supervision logits; uncertainty head.

![Feature-restoration illustration](assets/figures/Figure%202.png)

## Repository layout

```text
phydiff-unet-original-code/
├── README.md
├── notebooks/
│   ├── Unet_Wavelet_SIRST(1).ipynb         # original training notebook, unchanged
│   └── Experiements_3D(1).ipynb           # original tests/plots notebook, unchanged
├── src/phydiff_unet/
│   ├── model.py                           # verbatim model class cells
│   ├── losses_metrics.py                  # verbatim loss/metric cells
│   ├── notebook_evaluation.py             # verbatim result, heatmap, feature map functions
│   └── comparisons.py                     # verbatim 3D, 2D, ROC and FPS functions
├── scripts/
│   ├── predict.py                         # NEW optional inference adapter
│   ├── export_torchscript.py              # NEW optional TorchScript export adapter
│   └── run_comparisons.py                 # NEW optional plotting adapter
├── assets/figures/                       # Figures 1–8, original PNG bytes
├── paper/Elsevier_Small_Object(5).pdf    # supplied manuscript
├── docs/CODE_MAP.md
├── docs/REPRODUCIBILITY.md
├── tests/
└── .github/workflows/tests.yml
```

## Install

Recommended: **Google Colab** (the original training notebook includes Colab-only imports and absolute `/content` paths). For the standalone extracted modules and adapters:

```bash
python -m pip install -e '.[notebooks,test]'
pytest -q
```

## Train the original model

Open [`notebooks/Unet_Wavelet_SIRST(1).ipynb`](notebooks/Unet_Wavelet_SIRST(1).ipynb) in Colab and follow its cells **as written**. You must supply the author's original zipped dataset in the path expected by the notebook. The training cell saves the best state dictionary as `my_checkpoint.pth`.

Notebook-specific settings: 256x256 grayscale, `UnetBlocks(2)`, 100 epochs, Adam LR=3e-4 and weight decay=1e-4, batch size 1, and the exact loss/deep-supervision expression documented in the notebook. Refer to [code map](docs/CODE_MAP.md) and [reproducibility notes](docs/REPRODUCIBILITY.md).

## Predict with your checkpoint (optional new adapter)

```bash
python scripts/predict.py --checkpoint checkpoints/my_checkpoint.pth --image data/example.png --out prediction.png --probability-out probability.png
```

No pretrained checkpoint is bundled. Prediction uses the **original unmodified `UnetBlocks` implementation** at 256x256 input resolution.

## Export your original model to TorchScript

```bash
python scripts/export_torchscript.py --checkpoint checkpoints/my_checkpoint.pth --out Ours.pt
```

This new adapter traces the unchanged `UnetBlocks` in evaluation mode at 256×256. Direct `torch.jit.script` is not supported by one optional branch in the original PPE class; the architecture is not altered.

## Original 3D comparisons, ROC, 2D masks, and FPS

![Original 3D multi-model visualization](assets/figures/Figure%203.png)

![Original bounding-box comparison](assets/figures/Figure%204.png)

To run the original notebook-based comparisons, provide all five TorchScript model files: `ACM.pt`, `ALCNet.pt`, `DNA-NET.pt`, `SWIN-UNET.pt`, `Ours.pt`. The source notebooks hardcode these under `/content/`. Their original model-channel specification is `[3, 3, 1, 1, 1]` and the notebook relies on pre-existing image/mask test data.

A **new optional adapter** can call the **unchanged original plotting functions** with file paths you provide:

```bash
python scripts/run_comparisons.py --images data/test/images --masks data/test/masks \
  --models ACM.pt ALCNet.pt DNA-NET.pt SWIN-UNET.pt Ours.pt --mode roc
# Replace --mode roc with 3d, 2d or latency for other original plotting functions.
```

## Original reported figures (not regenerated)

| Figure | What it shows |
|---|---|
| 1 | PhyDiff U-Net architecture and PPE |
| 2 | Noise-guided feature restoration / FRE |
| 3 | Original 3D surface comparison |
| 4 | Original 2D detection / bounding-box comparison |
| 5 | Original FRE ablation visualization |
| 6 | Original physics-prior maps |
| 7 | Original ROC curves on three datasets |
| 8 | Original accuracy vs. FLOPs scatter plots |

![Original ROC curves](assets/figures/Figure%207.png)

![Original accuracy versus FLOPs](assets/figures/Figure%208.png)

These original uploaded plots are displayed as supplied. The source experiments notebook covers 3D, 2D, latency/FPS and ROC calculations; the datasets, checkpoints, benchmark metric tables, and original Figure 8 plot-generation data are not in the supplied upload.

## Code fidelity and limitations

See **[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** for explicit manuscript-versus-notebook differences; they have **not been corrected or changed**. Tests check verbatim extraction and a CPU model forward pass, not any published score.

## Citation

Please cite the accompanying *PhyDiff U-Net* manuscript and use the final accepted journal citation when available. Bibliographic author details, DOI, and final publication date have deliberately not been invented here.

## Licensing

No open-source license has been selected on the author's behalf. A license file can be added after the authors agree on release terms.
