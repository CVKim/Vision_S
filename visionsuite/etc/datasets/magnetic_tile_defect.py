import os
import os.path as osp 
import glob 
import numpy as np
import cv2
from shutil import copyfile
from tqdm import tqdm


input_dir = '/HDD/datasets/public/defects/sources/Surface-Defect-Detection-master/Magnetic-Tile-Defect'
output_dir = '/HDD/datasets/public/defects/magnetic_tile_defects'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

mask_output_dir = '/HDD/datasets/public/defects/magnetic_tile_defects/masks'
if not osp.exists(mask_output_dir):
    os.mkdir(mask_output_dir)

image_output_dir = '/HDD/datasets/public/defects/magnetic_tile_defects/images'
if not osp.exists(image_output_dir):
    os.mkdir(image_output_dir)

folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
label2idx = {}
for folder in folders:
    label = folder.split("_")[-1]
    
    _mask_output_dir = osp.join(mask_output_dir, label)
    if not osp.exists(_mask_output_dir):
        os.mkdir(_mask_output_dir)
    
    _image_output_dir = osp.join(image_output_dir, label)
    if not osp.exists(_image_output_dir):
        os.mkdir(_image_output_dir)
        
    if label not in label2idx:
        idx = len(label2idx) + 1
        label2idx[label] = idx

    gt_files = glob.glob(osp.join(input_dir, folder, "Imgs/*.png"))

    for gt_file in tqdm(gt_files, desc=folder):
        img_file = gt_file.replace('.png', '.jpg')
        
        assert osp.exists(img_file), ValueError(f'There is no such image: {img_file}')
        assert osp.exists(gt_file), ValueError(f'There is no such gt: {gt_file}')
        
        gt = cv2.imread(gt_file, 0)

        mask = np.zeros(gt.shape)
        
        gt = (gt != 0)
        mask += gt.astype(np.uint8)*idx
        
        cv2.imwrite(osp.join(_mask_output_dir, osp.split(gt_file)[-1]), mask)
        copyfile(img_file, osp.join(_image_output_dir, osp.split(img_file)[-1]))
        
        
        
        
        
        
        

    
    