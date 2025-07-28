'''
{
    "label_type": "cla",
    "source": "labelset",
    "classes": [
        {
            "name": "class1",
            "color": "rgba(248, 126, 172, 1)"
        },
        {
            "name": "class2",
            "color": "rgba(167, 238, 62, 1)"
        }
    ],
    "data": [
        {
            "fileName": "cla_01.jpg",
            "set": "train",
            "classLabel": "class1",
            "regionLabel": [],
            "retestset": 0,
            "rotation_angle": 0,
            "width": 1024,
            "height": 1024
        },
        {
            "fileName": "cla_02.jpg",
            "set": "test",
            "classLabel": "class2",
            "regionLabel": [],
            "retestset": 0,
            "rotation_angle": 0,
            "width": 1024,
            "height": 1024
        }
    ]
}
'''

import os
import json
import os.path as osp
from glob import glob 
import random 
from tqdm import tqdm 
from PIL import Image

ratio = 0.0
input_dir = '/DeepLearning/research/data/unittests/unit_cost_test/split_cls_tobacco_benchmark'
output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/neurocle/split_cls_tobacco_benchmark'

width = None
height = None

if not osp.exists(output_dir):
    os.mkdir(output_dir)

val_images = list(map(str, list(range(0, 0))))

os.makedirs(output_dir, exist_ok=True)

# labelme_train_imgs = glob(osp.join(input_dir, "train/**/**.bmp"))
# labelme_val_imgs = glob(osp.join(input_dir, "val/**/**.bmp"))
labelme_val_imgs = glob(osp.join(input_dir, "test/**/**.bmp"))

neuro_anns = {
    "label_type": 'cla',
    "source": "labelset",
    'classes': [],
    "data": [],
}

classes = []
# for labelme_imgs in [(labelme_train_imgs, 'train'), (labelme_val_imgs, 'val')]:
for labelme_imgs in [(labelme_val_imgs, 'val')]:
    mode = labelme_imgs[1]
    for labelme_img in tqdm(labelme_imgs[0], desc=mode):
        img_file = osp.basename(labelme_img)
        filename = osp.splitext(img_file)[0]
        
        # if filename not in '124071716355805_1_image_1_SOO_0':
        #     continue
            
        img_file = filename + '.png'
        
        val = False
        if filename in val_images:
            val = True
        
        with Image.open(labelme_img) as img:
            width, height = img.size
        
        _class = osp.split(labelme_img)[0].split('/')[-1]
        if _class not in classes:
            classes.append(_class)
        
        data = {
            'fileName': img_file, 
            "set": 'train' if mode == 'train' else 'test',
            "classLabel": _class,
            "regionLabel": [],
            'retestset': 0,
            'rotation_angle': 0,
            'width': width,
            'height': height
        }
        
        neuro_anns['data'].append(data)
            
        if val:
            data = {
                'fileName': str(int(filename) + 3835)+ '.png', 
                "set": "test",
                "classLabel": _class,
                "regionLabel": [],
                'retestset': 0,
                'rotation_angle': 0,
                'width': width,
                'height': height
            }
            
            neuro_anns['data'].append(data)
                
    
            
colors = ['rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          'rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          'rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          'rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          'rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          'rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          'rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)',
          ]

for label, color in zip(classes, colors[:len(classes)]):
    neuro_anns['classes'].append({"name": label, "color": color})
                    
# with open(osp.join(output_dir, 'neuro.json'), 'w') as json_file:
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
    