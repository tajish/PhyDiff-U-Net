"""Check exact source preservation, not model accuracy or the paper's reported results."""
from pathlib import Path
import json
import importlib
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
TRAIN=json.loads((ROOT/'notebooks'/'Unet_Wavelet_SIRST(1).ipynb').read_text())
EXPERIMENTS=json.loads((ROOT/'notebooks'/'Experiements_3D(1).ipynb').read_text())
MODEL_CELLS=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
METRIC_CELLS=[50, 52, 54, 55, 57, 59, 60, 61, 62]
EVAL_CELLS=[70, 74, 75, 78, 79]
COMP_CELLS=[17, 20, 25, 31]

def verify(module_file, notebook, ids):
    code=(ROOT/module_file).read_text()
    for cid in ids:
        marker=f'# >>> ORIGINAL NOTEBOOK CELL {cid} START (UNMODIFIED)\n'
        end=f'\n# <<< ORIGINAL NOTEBOOK CELL {cid} END (UNMODIFIED)'
        content=code.split(marker,1)[1].split(end,1)[0]
        assert content == ''.join(notebook['cells'][cid]['source']).rstrip('\n')

def test_verbatim_architecture():
    verify('src/phydiff_unet/model.py',TRAIN,MODEL_CELLS)

def test_verbatim_metrics_and_losses():
    verify('src/phydiff_unet/losses_metrics.py',TRAIN,METRIC_CELLS)

def test_verbatim_evaluation():
    verify('src/phydiff_unet/notebook_evaluation.py',TRAIN,EVAL_CELLS)

def test_verbatim_comparisons():
    verify('src/phydiff_unet/comparisons.py',EXPERIMENTS,COMP_CELLS)

def test_notebook_cell_counts():
    assert len(TRAIN['cells'])==81
    assert len(EXPERIMENTS['cells'])==33

def test_original_model_output_shapes_and_loss():
    import torch
    from phydiff_unet.model import UnetBlocks
    from phydiff_unet.losses_metrics import HeteroscedasticBCELoss
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model=UnetBlocks(2).eval()
    with torch.no_grad():
        out=model(torch.rand(1,1,256,256))
    assert len(out)==7
    assert out[0].shape==out[1].shape==(1,1,256,256)
    assert out[2].shape==out[3].shape==(1,1,32,32)
    assert all(z.shape==(1,1,256,256) for z in out[4:])
    val=HeteroscedasticBCELoss(num_samples=1,uncertainty_weight=1e-4)(out[0],out[1],torch.zeros_like(out[0]))
    assert torch.isfinite(val)

def test_original_figure_and_notebook_sha256():
    import hashlib
    for line in (ROOT/'docs/SOURCE_SHA256.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        expected, rel = line.split('  ',1)
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected

def test_trace_can_export_original_model():
    import torch
    from phydiff_unet.model import UnetBlocks
    torch.set_num_threads(2)
    model=UnetBlocks(2).eval()
    traced=torch.jit.trace(model,torch.randn(1,1,256,256),check_trace=False)
    with torch.no_grad():
        values=traced(torch.rand(1,1,256,256))
    assert len(values)==7
    assert values[0].shape==(1,1,256,256)
