import argparse
import os.path as osp
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import datasets_to_c, get_cluster_acc

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

def _parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='folder')
    parser.add_argument('--phis', type=str, default=["clipvitL14", "dinov2"], 
                                    nargs='+', help="Representation spaces to evaluate TURTLE", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 
                                     'clipvitB16', 'clipvitL14', 'dinov2'])
    # parser.add_argument('--root_dir', default='data')
    parser.add_argument('--root_dir', type=str, default='/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/crop/offset_auto')
    # parser.add_argument('--root_dir', type=str, default='/HDD/datasets/projects/ctr/BOOT_CURL_INSIDE')
    parser.add_argument('--ckpt_path', default='/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/crop/offset_auto/task_checkpoints/2space/clipvitL14_dinov2/folder/turtle_clipvitL14_dinov2_innerlr0.001_outerlr0.001_T6000_M100_coldstart_gamma10.0_bs1000_seed42_cluster2.pt')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = _parse_args()

    # Load pre-computed representations 
    Zs_val = [np.load(ROOT / f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    y_gt_val = np.load(ROOT / f"{args.root_dir}/labels/{args.dataset}_val.npy")

    print(f'Load dataset {args.dataset}')
    print(f'Representations of {args.phis}: ' + ' '.join(str(Z_val.shape) for Z_val in Zs_val))

    C = datasets_to_c[args.dataset]
    feature_dims = [Z_val.shape[1] for Z_val in Zs_val]
    
    # Task encoder
    task_encoder = [nn.Linear(d, C).to(args.device) for d in feature_dims] 
    ckpt = torch.load(args.ckpt_path)
    for task_phi, ckpt_phi in zip(task_encoder, ckpt.values()):
        task_phi.load_state_dict(ckpt_phi)

    # Evaluate clustering accuracy
    label_per_space = [F.softmax(task_phi(torch.from_numpy(Z_val).to(args.device)), dim=1) for task_phi, Z_val in zip(task_encoder, Zs_val)] # shape of (N, K, C)
    labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, C)

    y_pred = labels.argmax(dim=-1).detach().cpu().numpy()
    
    from dataset_preparation.data_utils import get_datasets
    
    _, val_dataset = get_datasets(args.dataset, None, args.root_dir)
    
    for idx, (val, pred) in enumerate(zip(val_dataset, y_pred)):
        _output_dir = osp.join(args.root_dir, f'outputs/{C} clusters')
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
            
        _output_dir = osp.join(_output_dir, str(pred))
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
        
        val[0].save(osp.join(_output_dir, f'{idx}.png'))
    
    cluster_acc, _ = get_cluster_acc(y_pred, y_gt_val)
    
    phis = '_'.join(args.phis)
    print(f'{args.dataset:12}, {phis:20}, Number of found clusters {len(np.unique(y_pred))}, Cluster Acc: {cluster_acc:.4f}')