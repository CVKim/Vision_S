import os
import json
import os.path as osp
from glob import glob 
import random 
from tqdm import tqdm 
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, ElementTree

project_ele = Element('Project')

SubElement(project_ele, 'Version').text = '1.0'
SubElement(project_ele, 'Type').text = 'Classification'
SubElement(project_ele, 'SpecificType').text = 'Developer'
SubElement(project_ele, 'ModifiedDate').text = '2025-06-26 15:59:45'

class2index = {'BOHEM_1': 0, 'ESSE_0.5': 1, 'FIIT_CHANGE': 2, 'MIX_BLU': 3, 'RAISON_FB': 4,
               'SOO_0.5': 5, 'BOHEM_3': 6, 'ESSE_1': 7, 'FIIT_UP': 8, 'MIX_ICEAN': 9, 'SOO': 10,
               'BOHEM_6': 11, 'ESSE_GOLD': 12, 'MIX_BANG': 13, 'NGP': 14, 'SOO_0.1': 15
            }
# output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/sage/split_cls_tobacco_benchmark/train.srproj'
output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/sage/split_cls_tobacco_benchmark/test.srproj'
image_dir = '/DeepLearning/research/data/unittests/unit_cost_test/split_cls_tobacco_benchmark'

class_group_ele = SubElement(project_ele, 'ClassGroup')
SubElement(class_group_ele, 'NumberOfClasses').text = str(len(class2index))
colors = [-1048576, -256, -7155632, -16731920, -1872887, -7156260,
          -1234, -34123, -11231231, -2323123, -123495, -3496434,
          -692349, -3592359, -234923, -129569]
for idx, class_name in enumerate(class2index.keys()):
    class_ele = SubElement(class_group_ele, 'Class')
    SubElement(class_ele, 'Name').text = class_name
    SubElement(class_ele, 'Color').text = str(colors[idx])

# img_files_train = glob(osp.join(image_dir, "train/*/*.bmp"))
# img_files_val = glob(osp.join(image_dir, "val/*/*.bmp"))
img_files_val = glob(osp.join(image_dir, "test/*/*.bmp"))
image_group_ele = SubElement(project_ele, 'ImageGroup')
# SubElement(image_group_ele, 'NumberOfImages').text = str(len(img_files_train) + len(img_files_val))
SubElement(image_group_ele, 'NumberOfImages').text = str(40)

count = 0
# for idx, labelme_imgs in tqdm(enumerate([(img_files_train, 'train'), (img_files_val, 'val')])):
for idx, labelme_imgs in tqdm(enumerate([(img_files_val, 'val')])):
    mode = labelme_imgs[1]
    for labelme_img in tqdm(labelme_imgs[0], desc=mode):
        filename = osp.split(osp.splitext(labelme_img)[0])[-1]
        _class = osp.split(labelme_img)[0].split('/')[-1]

        # if filename != '3103':
        #     continue
        
        with Image.open(labelme_img) as img:
            width, height = img.size
        
        image_ele = SubElement(image_group_ele, 'Image')
        count += 1
        # SubElement(image_ele, 'Path').text = f'\\\\aiv1' + image_dir.replace("/", "\\") + f"\\{mode}\\{_class}\\{filename}.bmp"
        SubElement(image_ele, 'Path').text = f'\\\\aiv1' + image_dir.replace("/", "\\") + f"\\test\\{_class}\\{filename}.bmp"
        SubElement(image_ele, 'Width').text = str(width)
        SubElement(image_ele, 'Height').text = str(height)
        SubElement(image_ele, 'SplitState').text = 'Training' if mode == 'train' else 'Validation'
        SubElement(image_ele, 'ClassIndexOfLabel').text = str(class2index[_class])
        
tree = ElementTree(project_ele)
tree.write(osp.join(output_dir), encoding='utf-8', xml_declaration=True)
            
    