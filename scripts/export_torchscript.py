"""Adapter for exporting an original model checkpoint into a TorchScript .pt file."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from phydiff_unet.model import UnetBlocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='Ours.pt')
    args = parser.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetBlocks(2).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev, weights_only=True), strict=True)
    model.eval()
    # The original PPE class defines high_pass only for the optional branch,
    # so torch.jit.script fails. Trace the *unchanged* eval model at 256x256.
    example = torch.randn(1, 1, 256, 256, device=dev)
    traced = torch.jit.trace(model, example, check_trace=True)
    traced.save(args.out)
    print('Saved TorchScript model to', args.out)

if __name__ == '__main__':
    main()
