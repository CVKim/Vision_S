import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
import random

from data_utils import get_dataloaders

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='labelme')
    parser.add_argument('--phis', type=str, default='dinov2', help="Representation spaces to run TURTLE", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 
                                     'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--input-dir', type=str, default='/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset')
    # parser.add_argument('--output-dir', default='/HDD/etc/curation/tenneco/embeddings')
    parser.add_argument('--input-dir', type=str, default='/HDD/datasets/projects/mr/split_dataset')
    parser.add_argument('--output-dir', default='/HDD/etc/curation/mr/embeddings')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)


def get_features(dataloader, model, device):
    all_features = []
    filenames = []
    with torch.no_grad():
        for x, filename in tqdm(dataloader):
            features = model(x.to(device))
            all_features.append(features.detach().cpu())
            filenames.extend(filename)

    return torch.cat(all_features).numpy(), filenames


phi_to_name = {'clipRN50': 'RN50', 'clipRN101': 'RN101', 'clipRN50x4': 'RN50x4', 'clipRN50x16': 'RN50x16', 'clipRN50x64': 'RN50x64',
                   'clipvitB32': 'ViT-B/32', 'clipvitB16': 'ViT-B/16', 'clipvitL14': 'ViT-L/14'}

def run(args=None):
    args = _parse_args(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.phis == 'dinov2':
        torch.hub.set_dir(os.path.join(args.output_dir, "checkpoints/dinov2"))
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
        model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        preprocess = None
    # else:
    #     ckpt_dir = os.path.join(args.root_dir, "checkpoints/clip")
    #     model, preprocess = clip.load(phi_to_name[args.phis], device=device, download_root=ckpt_dir)
    #     model.eval()
    #     print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    #     model = model.encode_image
    #     preprocess.transforms[2] = _convert_image_to_rgb
    #     preprocess.transforms[3] = _safe_to_tensor
    
    trainloader, valloader = get_dataloaders(args.dataset, preprocess, args.batch_size, args.input_dir)
    feats_train, filenames_train = get_features(trainloader, model, device)
    # feats_val, filenames_val = get_features(valloader, model, device)

    representations_dir = f"{args.output_dir}/representations/{args.phis}"
    if not os.path.exists(representations_dir):
        os.makedirs(representations_dir)

    np.save(f'{representations_dir}/{args.dataset}_train.npy', feats_train)
    # np.save(f'{representations_dir}/{args.dataset}_val.npy', feats_val)


    with open(f'{representations_dir}/{args.dataset}_train_filenames.txt', 'w') as f:
        for path in filenames_train:
            f.write(f"{path}\n")
            
    # with open(f'{representations_dir}/{args.dataset}_val_filenames.txt', 'w') as f:
    #     for path in filenames_val:
    #         f.write(f"{path}\n")

if __name__ == '__main__':
    run()
