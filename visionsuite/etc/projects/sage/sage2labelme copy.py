import os 
import os.path as osp 
from glob import glob 
import json 
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from visionsuite.utils.dataset.formats.labelme.utils import init_labelme_json, add_labelme_element

orders = ['1', '2', '3']
case = '2nd'
defects = ['오염', '경계성', '딥러닝', 'repeated_ng', 'repeated_ok']
w, h = 1120, 768
for defect in defects:
    img_dir = f'/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/{case}/neuro/{defect}'
    input_dir = f'/DeepLearning/etc/sage/{case}/{defect}'
    output_dir = f'/HDD/etc/repeatablility/talos2/{case}/benchmark/sage/{defect}'

    os.makedirs(output_dir, exist_ok=True)


    for order in orders:
        
        img_files = glob(osp.join(img_dir, order, '*.bmp'))

        csv_files = glob(osp.join(input_dir, order, '*-objects.csv'))
        print(csv_files)
        # assert len(csv_files) == 1, f'There are {len(csv_files)} at {input_dir, order}: {csv_files}'
        df = pd.read_csv(csv_files[0], encoding='utf-8')

        for img_file in tqdm(img_files, desc=defect):
            filename = osp.split(osp.splitext(img_file)[0])[-1]
            
            _labelme = init_labelme_json(filename + '.bmp', w, h)
            
            for idx, row in df.iterrows():
                file_path = row['file_path'].split("\\")[-1].split(".")[0]
                file_path_filename = osp.split(osp.splitext(file_path)[0])[-1]
                
                if filename == file_path_filename:
                    
                    
                    _labelme = add_labelme_element(_labelme, shape_type='polygon', 
                                            label=row['class_name'], 
                                            points=[[row['bbox_left'] + 220, row['bbox_top'] + 60], 
                                                    [row['bbox_left'] + 220 + row['bbox_width'], row['bbox_top'] + 60], 
                                                    [row['bbox_left'] + 220 + row['bbox_width'], row['bbox_top'] + 60 + row['bbox_height']],
                                                    [row['bbox_left'] + 220, row['bbox_top'] + 60 + row['bbox_height']], 
                                                    ]
                                        )


            if order == '1':
                _output_dir = osp.join(output_dir, f'exp/labels')
            else:     
                _output_dir = osp.join(output_dir, f'exp{order}/labels')

            os.makedirs(_output_dir, exist_ok=True)

            with open(os.path.join(_output_dir, filename + ".json"), "w") as jsf:
                json.dump(_labelme, jsf)
            
            
            
