import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import visionsuite.cores.masked_autoencoder.mae.models_mae as models_mae
from visionsuite.utils.download import download_weights_from_url

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, output_filename):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    plt.imshow(torch.clip((x[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())

    plt.title("original")

    plt.subplot(1, 4, 2)
    plt.imshow(torch.clip((im_masked[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title('masked')

    plt.subplot(1, 4, 3)
    plt.imshow(torch.clip((y[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title("reconstruction")

    plt.subplot(1, 4, 4)
    plt.imshow(torch.clip((im_paste[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title("reconstruction + visible")
    plt.axis('off')
    plt.savefig(output_filename)
        
img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]

# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)
url = 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth'
output_filename = '/HDD/weights/mae/mae_visualize_vit_large_75.pth'
download_weights_from_url(url, output_filename)

chkpt_dir = output_filename
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae, f'/HDD/etc/result.png')


url = 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth'
output_filename = '/HDD/weights/mae/mae_visualize_vit_large_ganloss.pth'
download_weights_from_url(url, output_filename)

chkpt_dir = output_filename
model_mae_gan = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

torch.manual_seed(2)
print('MAE with extra GAN loss:')
run_one_image(img, model_mae_gan, f'/HDD/etc/result_ganloss.png')