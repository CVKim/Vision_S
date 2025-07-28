import os 
import os.path as osp 
from glob import glob 
import json 
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from visionsuite.utils.dataset.formats.labelme.utils import init_labelme_json, add_labelme_element
from PIL import Image


def sage2labelme_seg():

    mode = 'train'
    img_dir = f'/DeepLearning/research/data/unittests/unit_cost_test/split_mr/{mode}'
    input_dir = f'/DeepLearning/research/data/unittests/unit_cost_test/saige/split_mr/results/{mode}_results/{mode}'
    output_dir = osp.join(input_dir, 'labelme')
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = glob(osp.join(img_dir, '*.bmp'))

    csv_files = glob(osp.join(input_dir, '*-objects.csv'))
    print(csv_files)
    # assert len(csv_files) == 1, f'There are {len(csv_files)} at {input_dir, order}: {csv_files}'
    df = pd.read_csv(csv_files[0], encoding='utf-8')

    for img_file in tqdm(img_files):
                
        with Image.open(img_file) as img:
            w, h = img.size
        
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        _labelme = init_labelme_json(filename + '.bmp', w, h)
        
        for idx, row in df.iterrows():
            file_path = row['file_path'].split("\\")[-1].split(".")[0]
            file_path_filename = osp.split(osp.splitext(file_path)[0])[-1]
            
            if filename == file_path_filename:
                
                
                _labelme = add_labelme_element(_labelme, shape_type='polygon', 
                                        label=row['class_name'], 
                                        points=[[row['bbox_left'], row['bbox_top']], 
                                                [row['bbox_left'] + row['bbox_width'], row['bbox_top']], 
                                                [row['bbox_left'] + row['bbox_width'], row['bbox_top'] + row['bbox_height']],
                                                [row['bbox_left'], row['bbox_top'] + row['bbox_height']], 
                                                ]
                                    )



        with open(os.path.join(output_dir, filename + ".json"), "w") as jsf:
            json.dump(_labelme, jsf)
        
def sage2labelme_det():
    mode = 'val'
    img_dir = f'/DeepLearning/research/data/unittests/unit_cost_test/split_interojo_dataset/{mode}'
    input_dir = f'/DeepLearning/research/data/unittests/unit_cost_test/saige/split_interojo_dataset/results/{mode}_results/{mode}'
    output_dir = osp.join(input_dir, 'labelme')
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = glob(osp.join(img_dir, '*.bmp'))

    csv_files = glob(osp.join(input_dir, '*-objects.csv'))
    print(csv_files)
    # assert len(csv_files) == 1, f'There are {len(csv_files)} at {input_dir, order}: {csv_files}'
    df = pd.read_csv(csv_files[0], encoding='utf-8')

    for img_file in tqdm(img_files):
                
        with Image.open(img_file) as img:
            w, h = img.size
        
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        _labelme = init_labelme_json(filename + '.bmp', w, h)
        
        for idx, row in df.iterrows():
            file_path = row['file_path'].split("\\")[-1].split(".")[0]
            file_path_filename = osp.split(osp.splitext(file_path)[0])[-1]
            
            if filename == file_path_filename:
                
                
                _labelme = add_labelme_element(_labelme, shape_type='rectangle', 
                                        label=row['class_name'], 
                                        points=[[row['bbox_left'], row['bbox_top']], 
                                                [row['bbox_left'] + row['bbox_width'], row['bbox_top']], 
                                                [row['bbox_left'] + row['bbox_width'], row['bbox_top'] + row['bbox_height']],
                                                [row['bbox_left'], row['bbox_top'] + row['bbox_height']], 
                                                ]
                                    )



        with open(os.path.join(output_dir, filename + ".json"), "w") as jsf:
            json.dump(_labelme, jsf)
            
if __name__ == '__main__':
    # sage2labelme_seg()
    sage2labelme_det()