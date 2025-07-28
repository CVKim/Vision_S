
import glob 
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

input_dir = '/Data/01.Image/ctr/2_CMFB/4_TEST/gt_bgra/gt_bgra'
output_dir = '/Data/01.Image/ctr/2_CMFB/4_TEST/gt_bgra/gt_bgra_mask'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.png'))

for img_file in tqdm(img_files):
    img = cv2.imread(img_file)
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    h, w, c = img.shape
    rgb_to_class = {
        (0, 0, 0): 0, 
        (255, 0, 0): 1,   # Red -> Class 0
        (0, 255, 0): 2,   # Green -> Class 1
        (0, 0, 255): 3    # Blue -> Class 2
    }
    
    class_image = np.zeros((h, w), dtype=np.uint8)
    
    for rgb, class_index in rgb_to_class.items():
        # RGB 값이 일치하는 부분에 대해 클래스 인덱스 할당
        mask = np.all(img == np.array(rgb), axis=-1)
        class_image[mask] = class_index

    
    cv2.imwrite(osp.join(output_dir, filename + '.png'), class_image)


