"""Run original 3D, 2D, ROC and latency comparison functions on five TorchScript checkpoints.

This is a new orchestration adapter: the imported notebook functions themselves are verbatim.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from PIL import Image
from phydiff_unet.comparisons import (compare_models_2d, compare_models_3d,
    compare_models_roc, calculate_latency_fps)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--models', nargs=5, required=True, metavar=('ACM', 'ALCNet', 'DNANet', 'SWINUnet', 'Ours'))
    ap.add_argument('--mode', choices=['3d','2d','roc','latency'], required=True)
    ap.add_argument('--num-images', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    names=['ACM','ALCNet','DNANet','SWINUnet','Ours']
    channels=[3,3,1,1,1] # exact channels from Experiements_3D original cell 18
    image_dir,mask_dir=Path(args.images),Path(args.masks)
    image_tensors=[];mask_tensors=[]
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
            continue
        q=mask_dir/p.name
        if not q.exists(): continue
        x=np.array(Image.open(p).convert('L'),copy=True)
        y=np.array(Image.open(q).convert('L'),copy=True)
        if x.shape != y.shape: raise ValueError('Mask and image dimensions differ: '+str(p))
        image_tensors.append(torch.from_numpy(x).float().unsqueeze(0)/255.)
        mask_tensors.append(torch.from_numpy(y).float().unsqueeze(0)/255.)
    if not image_tensors: raise RuntimeError('No matching image/mask filenames')
    if args.mode=='3d':
        compare_models_3d(image_tensors,mask_tensors,args.models,names,channels,device=args.device,num_images=args.num_images)
    elif args.mode=='2d':
        compare_models_2d(image_tensors,mask_tensors,args.models,names,channels,device=args.device,num_images=args.num_images)
    elif args.mode=='roc':
        compare_models_roc(image_tensors,mask_tensors,args.models,names,channels,device=args.device)
    else:
        print(calculate_latency_fps(image_tensors,args.models,names,channels,device=args.device))

if __name__=='__main__': main()
