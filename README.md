# PhyDiff U-Net

**A Physics-Informed Diffusion-Inspired Transformer for Infrared Small-Target Segmentation**

[**Manuscript**](paper/Elsevier_Small_Object%285%29.pdf) · [**Training notebook**](notebooks/Unet_Wavelet_SIRST%281%29.ipynb) · [**Evaluation and visualization notebook**](notebooks/Experiements_3D%281%29.ipynb)

## Overview

Infrared small-target segmentation is challenging because distant targets may occupy only a few pixels and exhibit weak contrast against complex backgrounds, including clouds, terrain, and reflections. Repeated feature downsampling can further weaken target responses, while background structures with similar intensity patterns may generate false detections.

**PhyDiff U-Net** combines physics-guided feature extraction, efficient Transformer attention, noise-perturbation-guided feature refinement, and uncertainty-aware segmentation. Thermal-contrast and frequency-domain priors guide the encoder an edge-aware efficient Transformer (EET) models contextual relationships at the bottleneck and a Feature Restoration Enhancement (FRE) module refines target features before boundary-aware decoding. The network produces a segmentation logit map and a pixel-wise log-variance map, with auxiliary outputs for deep supervision.

![PhyDiff U-Net architecture](assets/figures/Figure%201.png)

*Figure 1. Architecture of PhyDiff U-Net, showing the Physics Prior Extraction (PPE) modules, wavelet encoder–decoder, EET, FRE, deep supervision, and uncertainty head.*

## Method

### Physics Prior Extraction (PPE)

PPE combines two complementary infrared imaging cues:

- **Thermal-contrast prior:** a Gaussian point-spread-function response is compared with an annular neighborhood response to emphasize locally salient targets.
- **Frequency prior:** Haar discrete wavelet transform (DWT) features capture low- and high-frequency information. The wavelet features are projected, resized, and fused with the thermal prior.

The resulting physics-aware representation is used throughout the encoder and to condition feature refinement.

### Edge-Aware Efficient Transformer (EET)

At the bottleneck, EET combines separable self-attention for global context aggregation with edge-aware processing to retain boundary-sensitive information. Separable attention avoids the quadratic token-to-token attention matrix used in conventional self-attention.

### Feature Restoration Enhancement (FRE)

FRE introduces Gaussian perturbations to bottleneck features **during training** and refines them using the physics-prior representation. A learned residual is added to the clean bottleneck features. At inference, noise injection is disabled and refinement requires a single forward pass; the module is **not** a multi-step denoising diffusion probabilistic model.

![Feature Restoration Enhancement module](assets/figures/Figure%202.png)

*Figure 2. Noise-perturbation-guided feature refinement conditioned on physics-prior information.*

### Boundary-aware decoder and uncertainty estimation

The decoder uses inverse wavelet transform (IWT) for spatial reconstruction, spatial attention on encoder skip features, and Sobel-based boundary attention (SDA). The prediction head estimates segmentation logits and pixel-wise log-variance for heteroscedastic uncertainty modeling. Auxiliary heads provide intermediate segmentation predictions for deep supervision.

The model implementation returns:

```python
mean, log_var, bot_head, bot_ddpm, head2, head3, head4 = model(image)
```

## Results

The following are **results reported in the manuscript**, not measurements reproduced by running this repository. The evaluation uses Intersection over Union (IoU), normalized IoU (nIoU), probability of detection (Pd), and false-alarm rate (Fa).

| Dataset | IoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ (×10⁻⁶) |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 76.87% | 73.79% | 99.16% | 3.34 |
| NUDT-SIRST | 91.20% | 92.39% | 98.00% | 12.93 |
| IRSTS-1K | 65.75% | 65.98% | 92.83% | 16.63 |

*Source: manuscript, Section 5 and Table 2; detailed values are given in the accompanying discussion.*

The manuscript reports the following computational characteristics for PhyDiff U-Net on NUAA-SIRST:

| Parameters | FLOPs | Inference latency | Throughput |
|---:|---:|---:|---:|
| 3.16M | 5.732G | 6.605 ms/frame | 151.39 FPS |

*Source: manuscript, Table 6. Experimental setup: a single NVIDIA L4 GPU in Google Colab, as described in Section 4.3.*

### Qualitative comparisons

![3D segmentation response comparisons](assets/figures/Figure%203.png)

*Figure 3. Three-dimensional visualization of infrared images, ground-truth masks, and prediction responses from competing methods.*

![Target detection comparisons](assets/figures/Figure%204.png)

*Figure 4. Qualitative target-localization comparisons, including correct detections, missed targets, and false alarms.*

### Ablation and physics-prior visualization

![FRE ablation visualization](assets/figures/Figure%205.png)

*Figure 5. Feature response comparison with and without FRE.*

![Physics-prior visualization](assets/figures/Figure%206.png)

*Figure 6. Physics-prior representations generated by PPE.*

### ROC and accuracy–efficiency analysis

![ROC comparisons](assets/figures/Figure%207.png)

*Figure 7. ROC curves on IRSTS-1K, NUDT-SIRST, and NUAA-SIRST.*

![Accuracy versus FLOPs](assets/figures/Figure%208.png)

*Figure 8. IoU versus FLOPs for PhyDiff U-Net and the compared methods across the three benchmarks.*

## Datasets and experimental settings

Three infrared small-target benchmarks are evaluated in the manuscript:

| Dataset | Images | Type | Manuscript train/validation/test split |
|---|---:|---|---|
| NUAA-SIRST | 427 | Real | 90:5:5 |
| NUDT-SIRST | 1,327 | Synthetic | 80:10:10 |
| IRSTS-1K | 1,000 | Real | 80:10:10 |

The manuscript specifies 256 × 256 single-channel inputs, 100 training epochs, Adam with an initial learning rate of `3e-4`, weight decay `1e-4`, and batch size `1`. Its validation and test protocol is described in Section 4. The training notebook retains its own dataset paths and split logic; consult [reproducibility notes](docs/REPRODUCIBILITY.md) when reproducing dataset-specific experiments.

## Repository structure

```text
.
├── README.md
├── notebooks/
│   ├── Unet_Wavelet_SIRST(1).ipynb
│   └── Experiements_3D(1).ipynb
├── src/phydiff_unet/
│   ├── model.py
│   ├── losses_metrics.py
│   ├── notebook_evaluation.py
│   └── comparisons.py
├── scripts/
│   ├── predict.py
│   ├── export_torchscript.py
│   └── run_comparisons.py
├── assets/figures/
│   └── Figure 1.png ... Figure 8.png
├── paper/
│   └── Elsevier_Small_Object(5).pdf
├── docs/
│   ├── CODE_MAP.md
│   ├── REPRODUCIBILITY.md
│   └── SOURCE_SHA256.txt
├── tests/
├── .github/workflows/tests.yml
├── pyproject.toml
└── requirements.txt
```

The model, loss, metric, and comparison functions are available under `src/phydiff_unet/`. The notebooks contain the original Colab training and evaluation workflows. The scripts provide command-line entry points for checkpoint inference, TorchScript export, and comparison plotting.

## Installation

Python 3.10 or later and PyTorch are required. Install the repository from its root directory:

```bash
python -m pip install -e ".[notebooks,test]"
python -m pytest -q
```

Google Colab is recommended for running the training notebook, which uses Colab-specific paths and imports.

## Training

Open [`notebooks/Unet_Wavelet_SIRST(1).ipynb`](notebooks/Unet_Wavelet_SIRST%281%29.ipynb) in Google Colab. Configure the image and mask dataset locations used by the notebook, then run the training cells. The notebook instantiates `UnetBlocks(2)` and saves the selected model state dictionary to `my_checkpoint.pth`.

The notebook and manuscript differ in some training details, including FRE noise initialization, heteroscedastic loss sampling, and dataset splitting. These differences are recorded in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); the underlying model and training cells have not been modified.

## Inference

Given a trained `my_checkpoint.pth` state dictionary and an infrared image:

```bash
python scripts/predict.py --checkpoint checkpoints/my_checkpoint.pth --image data/example.png --out prediction.png --probability-out probability.png
```

The script loads `UnetBlocks(2)`, resizes the image to 256 × 256 grayscale, and saves a binary prediction mask. The probability image is optional. The model checkpoint must be supplied separately.

## TorchScript export

```bash
python scripts/export_torchscript.py --checkpoint checkpoints/my_checkpoint.pth --out Ours.pt
```

The export script traces the model at 256 × 256 in evaluation mode. The resulting `Ours.pt` can be used with the comparison functions.

## ROC curves, 3D visualization, and model comparisons

The comparison notebook contains the original visualization and measurement routines for 3D target responses, 2D target localization, ROC curves, and inference latency/FPS.

For example:

```bash
python scripts/run_comparisons.py --images data/test/images --masks data/test/masks --models ACM.pt ALCNet.pt DNA-NET.pt SWIN-UNET.pt Ours.pt --mode roc
```

Use `--mode 3d`, `--mode 2d`, or `--mode latency` for the other supported comparisons. Image and mask filenames must match. The notebooks and scripts require the appropriate dataset and model files; those inputs are not included in the repository.

## Reproducibility

The manuscript's reported results have not been independently recomputed from the repository contents alone. For implementation details and differences between the manuscript configuration and notebook code, see [CODE_MAP.md](docs/CODE_MAP.md) and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The [manuscript PDF](paper/Elsevier_Small_Object%285%29.pdf) is included for reference. Use the final bibliographic citation when publication details are available.
