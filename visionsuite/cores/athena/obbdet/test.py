import glob 
import os
import os.path as osp
import torch
import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from visionsuite.utils.visualizers.vis_obb import vis_obb
from athena.src.tasks.obbdetection.frameworks.pytorch.models.utils.nms import nms_rotated
from athena.src.tasks.obbdetection.frameworks.pytorch.models.utils.transforms import obb2poly_np_le90
from visionsuite.utils.helpers import letterbox, JsonEncoder


compare_mask = True
ml_framework = 'pytorch'
model_name='rtmdet'
backbone = 'large'
weights = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/outputs/outputs/OBBDETECTION/2024_10_11_16_05_36/train/weights/last.pt'
ckpt = torch.load(weights, map_location='cpu')
model = ckpt['model']
# model = de_parallel(model)
model.bbox_head.test_cfg = {
        'score_thr': 0.6,
        'iou_thr': 0.25,
        'max_per_img': 300,
        'agnostic_nms': False}

model.to('cuda:0')

compare_gt = True
iou_threshold = 0.7
width, height = 768, 768
max_dets = 50
nms_conf_threshold = 0.5
nms_iou_threshold = 0.25
classes = ['BOX']
idx2class = {idx: cls for idx, cls in enumerate(classes)}
_classes = ['BOX']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}
input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/sfaw'
json_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/sfaw'
output_dir = f'/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/tests/sfaw/{model_name}_{backbone}'
# input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/split_dataset_rich/val'
# json_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/split_dataset_rich/val'
# output_dir = f'/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/tests/rich/{model_name}_{backbone}'

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

results = {}
compare = {}
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file).astype(np.float32)
    img_h, img_w, img_c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    img, ratio, pad_values = letterbox(img, (width, height))
    
    torch_img = torch.from_numpy(np.transpose(img[np.newaxis, :], (0, 3, 1, 2))).to('cuda')
    model.forward = model.forward_export
    with torch.no_grad():
        preds = model(torch_img)
    pred = preds[0].detach().cpu()
    
    # start_angle = -90 / 180*np.pi
    # x, y, w, h, r = pred[:, :-1].unbind(-1)
    # _w = torch.where(w > h, w, h)
    # _h = torch.where(w > h, h, w)
    # r = torch.where(w > h, r, r + np.pi/2)
    # r = ((r - start_angle)%np.pi) + start_angle
    
    # pred[:, :-1] = torch.stack([x, y, _w, _h, r], dim=-1)
    
    dets, _ = nms_rotated(pred[:, :-1], pred[:, -1], iou_threshold=nms_iou_threshold)
    confs, labels = torch.max(dets[:, 5:], axis=1)

    idx2xyxys = {}
    for det, conf, cls in zip(dets[:max_dets].numpy(), confs[:max_dets].numpy(), labels[:max_dets].numpy()):
        if conf >= nms_conf_threshold:
            if cls not in idx2xyxys.keys():
                idx2xyxys[cls] = {'polygon': [], 'confidence': []}

            det[0] -= pad_values[0]
            det[1] -= pad_values[1]
            det[0] /= ratio[0]
            det[1] /= ratio[1]
            det[2] /= ratio[0]
            det[3] /= ratio[1]

            poly_points = det[np.newaxis, ...]
            poly_points = obb2poly_np_le90(poly_points)
            poly_points = poly_points[:, :8].reshape(4, 2)
            poly_points = np.round(poly_points, 2)

            idx2xyxys[cls]['polygon'].append([list(poly_point) for poly_point in poly_points])
            idx2xyxys[cls]['confidence'].append(conf)
  
    if _classes is not None:
        new_idx2xyxys = {}
        for idx, _cls in enumerate(_classes):
            for jdx, cls in enumerate(idx2class.values()):
                if cls == _cls:
                    new_idx2xyxys[idx] = idx2xyxys[jdx]
        
        idx2xyxys = new_idx2xyxys
        idx2class = _idx2class   
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
    results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_gt:
        _compare = vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold)
            
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4, cls=JsonEncoder)

if compare_gt:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))