from visionsuite.utils.dataset.formats.labelme.utils import *

import glob 
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

input_dir = '/Data/01.Image/ctr/2_CMFB/4_TEST/gt_bgra/gt_bgra_mask'

img_files = glob.glob(osp.join(input_dir, '*.png'))

for img_file in tqdm(img_files):
    img = cv2.imread(img_file, -1)   
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    h, w = img.shape
    
    _labelme = init_labelme_json(filename + ".png", w, h)
    _labelme = get_points_from_image(
                                img,
                                ['background', 'ballstud_s', 'taper_s', 'ballstud_head_s'],
                                [0, 0],
                                [0, 0],
                                _labelme,
                                50,
                            )
    
    with open(os.path.join(input_dir, filename + ".json"), "w") as jsf:
        json.dump(_labelme, jsf)