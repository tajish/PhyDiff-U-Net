# Original-code map (no architectural changes)

The model's implementation comes from **`Unet_Wavelet_SIRST(1).ipynb`**; the original notebooks are byte-for-byte copies of the supplied files.

| Source notebook cell | Source element | Verbatim module |
|---|---|---|
| 18 | `PG_VEM` (defined but not used by `UnetBlocks`) | `src/phydiff_unet/model.py` |
| 20 | `BottleneckDiffusionEnhancement` (FRE) | model.py |
| 22 | `UncertaintyHead` | model.py |
| 24 | `EdgeDetection` (defined, not instantiated in `UnetBlocks`) | model.py |
| 26 | `SobelBoundaryAttention` | model.py |
| 28 | `DWT` | model.py |
| 30 | `PhysicsPriorExtraction` | model.py |
| 32 | `IWT` | model.py |
| 34–46 | attention, normalization, transformer classes | model.py |
| 48 | `UnetBlocks`, seven-output model | model.py |
| 50–62 | exact original losses and metrics | `src/phydiff_unet/losses_metrics.py` |
| 63 | original optimizer, LR scheduler, heteroscedastic loss config | unchanged in notebook |
| 67 | original 100-epoch training loop | unchanged in notebook |
| 70, 74, 75, 78, 79 | results/heatmap/feature-map code | `src/phydiff_unet/notebook_evaluation.py` |

Original **`Experiements_3D(1).ipynb`** cells used:

| Cell | Function | Verbatim module |
|---|---|---|
| 17 | `compare_models_3d` | `src/phydiff_unet/comparisons.py` |
| 20 | connected components / centroid matching / `compare_models_2d` | comparisons.py |
| 25 | `calculate_latency_fps` | comparisons.py |
| 31 | `compare_models_roc` | comparisons.py |

Cells 27 and 20 of the experiment notebook overlap; the extracted module takes the earlier cell 20. The complete original notebook includes both, unmodified.

The original code assumes Colab paths under `/content`, 256 x 256 images for PhyDiff U-Net, `UnetBlocks(2)`, TorchScript checkpoints for baselines, and GPU availability in some experiment cells. Consult `docs/REPRODUCIBILITY.md` before running. The standalone imports and paths of the new helper scripts are packaging glue, *not changes to the original notebooks.*
