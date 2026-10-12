"""Separate CLI adapter. DOES NOT modify the original notebook/model.

Example:
 python scripts/predict.py --checkpoint checkpoints/my_checkpoint.pth --image data/example.png --out prediction.png
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from PIL import Image
from phydiff_unet.model import UnetBlocks


def main():
    parser = argparse.ArgumentParser(description='Predict using the exact original UnetBlocks model')
    parser.add_argument('--checkpoint', required=True, help='Original state_dict my_checkpoint.pth')
    parser.add_argument('--image', required=True, help='Grayscale input image')
    parser.add_argument('--out', default='prediction.png')
    parser.add_argument('--probability-out', help='Optional grayscale probability image')
    args = parser.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(args.image).convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    tensor = torch.tensor(np.asarray(image).copy(), device=dev, dtype=torch.float32)[None,None] / 255.
    model = UnetBlocks(2).to(dev)
    checkpoint = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    with torch.no_grad():
        mean, log_var, bot_head, bot_ddpm, head2, head3, head4 = model(tensor)
        probability = torch.sigmoid(mean)[0,0].cpu().numpy()
    Image.fromarray(((probability > 0.5) * 255).astype(np.uint8)).save(args.out)
    if args.probability_out:
        Image.fromarray(np.clip(probability*255,0,255).astype(np.uint8)).save(args.probability_out)
    print('Saved mask to', args.out)

if __name__ == '__main__':
    main()
