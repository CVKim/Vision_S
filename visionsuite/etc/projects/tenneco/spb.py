from glob import glob 
import os.path as osp 
import os 
import shutil
from tqdm import tqdm
import json 
import cv2

roi = [220, 60, 1340, 828]
input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset/train'
output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/sbp_dataset/tenneco'

if not osp.exists(output_dir):
    os.makedirs(output_dir)


img_files = glob(osp.join(input_dir, '*.bmp'))

for idx, img_file in tqdm(enumerate(img_files)):
    img = cv2.imread(img_file)
    img = img[roi[1]:roi[3], roi[0]:roi[2]]
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    json_file = osp.splitext(img_file)[0] + '.json'
    
    assert osp.exists(json_file), ValueError(f'There is no such json file: {json_file}')
    
    with open(json_file, 'r') as jf:
        anns = json.load(jf)
        
    anns['imagePath'] = f"{idx}.bmp"
    anns["imageHeight"] = roi[3] - roi[1]
    anns["imageWidth"] = roi[2] - roi[0]
    del anns['rois']

    empty_indexes = []
    for jdx, ann in enumerate(anns['shapes']):
        points = ann['points']
        del ann['bbox']
        if len(points) < 2:
            empty_indexes.append(jdx)
            continue 
        
        new_points = []
        for point in points:
            new_points.append([point[0] - roi[0], point[1] - roi[1]])
        ann['points'] = new_points
        
    for empty_index in reversed(empty_indexes):
        del anns['shapes'][empty_index]
        
    if len(anns['shapes']) == 0:
        anns = {
            "version": "4.0.0",
            "flags": {},
            "shapes": [],
            "imagePath": f"{idx}.bmp",
            "imageData": None,
            "imageHeight": roi[3] - roi[1],
            "imageWidth": roi[2] - roi[0]
            }
        
    data_dir = osp.join(output_dir, 'data')
    if not osp.exists(data_dir):
        os.mkdir(data_dir)
        
    cv2.imwrite(osp.join(data_dir, f'{idx}.bmp'), img)


    annotations_dir = osp.join(output_dir, 'annotations')
    if not osp.exists(annotations_dir):
        os.mkdir(annotations_dir)
        
    with open(osp.join(annotations_dir, f'{idx}.json'), 'w') as file:
        json.dump(anns, file)    
        
    
