import os.path as osp
import pandas as pd
from check_repeatability import run
import matplotlib.pyplot as plt

### --------------------------------------------------------------------------------------------------------
base_dir = '/DeepLearning/etc/neuro/1st'
dir_names = ['경계성']
cases = ['labelme']
filename_postfix = ''
rect_iou = True 
offset = 100

### ===================================
iou_thresholds = [0.05, 0.1, 0.2, 0.3]
area_thresholds = [20, 100, 150, 200]
figs = True 
vis = False
for case in cases:
    run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, 
        offset=offset, filename_postfix=filename_postfix, case=case)
