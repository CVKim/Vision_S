import os
import sys
import logging
import torch
import numpy as np

from os.path import splitext
from os import listdir
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp


class BasicDataset(Dataset):
    def __init__(self, unet_type, imgs_dir, masks_dir, scale=1):
        self.unet_type = unet_type
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.img_files = glob(osp.join(imgs_dir, "*.bmp"))
        logging.info(f'Creating dataset with {len(self.img_files)} examples')


    def __len__(self):
        return len(self.img_files)


    @classmethod
    def preprocess(cls, unet_type, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        if unet_type != 'v3':
            pil_img = pil_img.resize((newW, newH))
        else:
            new_size = int(scale * 640)
            pil_img = pil_img.resize((new_size, new_size))

        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans


    def __getitem__(self, i):
        img_file = self.img_files[i]
        mask_file = osp.join(self.masks_dir, osp.split(osp.splitext(img_file)[0])[-1] + '.bmp')

        assert osp.exists(img_file), ValueError(f'THere is no such image: {img_file}')
        assert osp.exists(mask_file), ValueError(f'THere is no such mask: {mask_file}')

        mask = Image.open(mask_file)
        img = Image.open(img_file)


        img = self.preprocess(self.unet_type, img, self.scale)
        mask = self.preprocess(self.unet_type, mask, self.scale)
        
        
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
