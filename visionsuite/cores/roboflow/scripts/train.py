import argparse 
import time
import os.path as osp 
import subprocess as sp
from visionsuite.cores.roboflow.ultralytics.train import train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    try:
        sp.run(['chmod', '-R', '777', args.output_dir])
        print(f"SUBPROCESS: {args.output_dir}")
    except Exception as error:
        print(f"Cannot chmod for {args.output_dir}: {error}")
        
    train(recipe_dir=args.recipe_dir)

    try:
        sp.run(['chmod', '-R', '777', args.output_dir])
        print(f"SUBPROCESS: {args.output_dir}")
    except Exception as error:
        print(f"Cannot chmod for {args.output_dir}: {error}")