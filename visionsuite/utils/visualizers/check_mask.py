import os.path as osp 
import glob 
import cv2 
import numpy as np

input_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/datasets/split_mask_patch_dataset/val/masks'

img_files = glob.glob(osp.join(input_dir, '*.jpg'))

for img_file in img_files:
    img = cv2.imread(img_file, 0)
    
    _img = np.unique(img)
    if 3. in list(_img):
        print(_img)
        print(img_file)
    

