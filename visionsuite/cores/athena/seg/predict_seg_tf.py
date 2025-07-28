import glob 
import os
import os.path as osp

import imgviz
import json
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from visionsuite.utils.visualizers.vis_seg import vis_seg
from athena.src.tasks.segmentation.frameworks.tensorflow.models.tf_model_v2 import TFModelv2


def get_mask_from_pred(pred, conf=0.5, contour_thres=10):
    mask = (pred > conf).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for contour in contours:
        if len(contour) < contour_thres:
            pass
        else:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].tolist()
            points.append(approx)

    return mask, points

compare_mask = False
weights = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/tf_deeplabv3plus_epochs200/train/weights/last_weights.h5'
model = TFModelv2(model_name='deeplabv3plus', backbone='efficientnetb3', backbone_weights='imagenet', backbone_trainable=True, 
                  batch_size=1, width=1120, height=768, channel=3, num_classes=4,
                  weights=weights)
blob_confidence = 0.8
classes = ["CHAMFER_MARK", "LINE", "MARK"]
idx2class = {idx: cls for idx, cls in enumerate(classes)}
_classes = ["CHAMFER_MARK", "LINE", "MARK"]
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}
input_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/val'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/val'
output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/tf_deeplabv3plus_epochs200/test/exp_conf{blob_confidence}'
roi = [220, 60, 1340, 828]

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

results = {}
compare = {}
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    
    if roi is not None:
        img = img[roi[1]:roi[3], roi[0]:roi[2]]
    
    preds = model(img[np.newaxis, :])[0].numpy()
    
    idx2xyxys = {}
    for idx in idx2class.keys():
        mask, points = get_mask_from_pred(preds[:, :, idx + 1], conf=blob_confidence)
    
        for _points in points:
            for _point in _points:
                _point[0] += roi[0]
                _point[1] += roi[1]
        
            if idx not in idx2xyxys:
                idx2xyxys[idx] = {'polygon': []}
            
            idx2xyxys[idx]['polygon'].append(_points)
        
    # if _classes is not None:
    #     new_idx2xyxys = {}
    #     for idx, _cls in enumerate(_classes):
    #         for jdx, cls in enumerate(classes):
    #             if cls == _cls:
    #                 new_idx2xyxys[idx] = idx2xyxys[jdx]
        
    #     idx2xyxys = new_idx2xyxys
    #     idx2class = _idx2class    
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1]
    results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_mask:
        _compare = vis_seg(img_file, idx2xyxys, idx2class, output_dir, color_map, seg_type='semantic', 
                           json_dir=json_dir, compare_mask=compare_mask)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_seg(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir=json_dir, compare_mask=compare_mask)
    
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

if compare_mask:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
        
    df_compare_pixel = df_compare.loc['diff_pixel'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_pixel.csv'))
    
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))