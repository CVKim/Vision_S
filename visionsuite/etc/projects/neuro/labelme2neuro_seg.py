import os
import json
import os.path as osp
from glob import glob 
import random 
from tqdm import tqdm 
import cv2

ratio = 0.0
input_dir = '/DeepLearning/research/data/unittests/unit_cost_test/split_mr'
output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/neurocle/split_mr'

width = None
height = None

if not osp.exists(output_dir):
    os.mkdir(output_dir)

val_images = list(map(str, list(range(0, 0))))

os.makedirs(output_dir, exist_ok=True)

labelme_train_jsons = glob(osp.join(input_dir, "train/*.json"))
# labelme_val_jsons = glob(osp.join(input_dir, "val/*.json"))
labelme_val_jsons = glob(osp.join(input_dir, "test/*.json"))

neuro_anns = {
    "label_type": 'seg',
    "source": "labelset",
    'classes': [],
    "data": [],
}

classes = []
for labelme_jsons in [(labelme_train_jsons, 'train'), (labelme_val_jsons, 'val')]:
    mode = labelme_jsons[1]
    for labelme_json in tqdm(labelme_jsons[0], desc=mode):
        json_file = osp.basename(labelme_json)
        filename = json_file.split('.')[0]
        
        # if filename != '0_coaxial_20240920101702728':
        #     continue
            
        img_file = filename + '.png'
        
        val = False
        if filename in val_images:
            val = True
        
        with open(labelme_json, 'r') as jf:
            labelme_anns = json.load(jf)
            
        width, height = labelme_anns['imageWidth'], labelme_anns['imageHeight']
        
        data = {
            'fileName': img_file, 
            "set": 'train' if mode == 'train' else 'test',
            'retest': 1,
            "classLabel": "",
            "regionLabel": [],
            'retestset': 0,
            'rotation_angle': 0,
            'width': width,
            'height': height
        }
            
        for labelme_ann in labelme_anns['shapes']:
            label = labelme_ann['label']
            if label not in classes:
                classes.append(label)
            points = labelme_ann['points']
            
            if labelme_ann['shape_type'] == 'polygon' and len(points) > 2:
                
                data['regionLabel'].append({
                    "className": label,
                    "type": "PolyLine",
                    "strokeWidth": 5,
                    "points": [[int(x) for x in pair] for pair in points]
                })  
                
        neuro_anns['data'].append(data) 
        
        if val:
            data = {
                'fileName': str(int(filename) + 3835)+ '.png', 
                "set": "test",
                "classLabel": "",
                "regionLabel": [],
                'retestset': 0,
                'rotation_angle': 0,
                'width': width,
                'height': height
            }
            with open(labelme_json, 'r') as jf:
                labelme_anns = json.load(jf)['shapes']
                
            for labelme_ann in labelme_anns:
                label = labelme_ann['label']
                if label not in classes:
                    classes.append(label)
                points = labelme_ann['points']
                
                data['regionLabel'].append({
                    "className": label,
                    "type": "PolyLine",
                    "strokeWidth": 5,
                    "points": [[int(x) for x in pair] for pair in points]
                })  
                    
            neuro_anns['data'].append(data)          
            
            
colors = ['rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)']

for label, color in zip(classes, colors[:len(classes)]):
    neuro_anns['classes'].append({"name": label, "color": color})
                    
# with open(osp.join(output_dir, 'train.json'), 'w') as json_file:
with open(osp.join(output_dir, 'test.json'), 'w') as json_file:
    json.dump(neuro_anns, json_file, ensure_ascii=False, indent=4)
    
    
# import os.path as osp 
# from glob import glob 
# from shutil import copyfile

# output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/neuro/renamed_val_images'
# img_files = glob(osp.join('/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/neuro/val_iamges', '*.bmp'))

# for img_file in img_files:
#     filename = osp.split(osp.splitext(img_file)[0])[-1]
#     copyfile(img_file, osp.join(output_dir, str(int(filename) + 3835)+ '.bmp'))
    