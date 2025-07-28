import os
import json
import os.path as osp
from glob import glob 
import random 
from tqdm import tqdm 
from xml.etree.ElementTree import Element, SubElement, ElementTree

roi = [220, 60, 1340, 828]
project_ele = Element('Project')

SubElement(project_ele, 'Version').text = '1.0'
SubElement(project_ele, 'Type').text = 'Segmentation'
SubElement(project_ele, 'SpecificType').text = 'Developer'
SubElement(project_ele, 'ModifiedDate').text = '2025-06-26 15:59:45'

class2index = {'STABBED': 0, 'DUST': 1, 'EDGE_STABBED': 2}
# output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/sage/split_mr/train.srproj'
output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/sage/split_mr/test.srproj'
json_dir = '/DeepLearning/research/data/unittests/unit_cost_test/split_mr'

class_group_ele = SubElement(project_ele, 'ClassGroup')
SubElement(class_group_ele, 'NumberOfClasses').text = str(len(class2index))
colors = [-1048576, -256, -7155632, -16731920, -1872887, -7156260]
for idx, class_name in enumerate(class2index.keys()):
    class_ele = SubElement(class_group_ele, 'Class')
    SubElement(class_ele, 'Name').text = class_name
    SubElement(class_ele, 'Color').text = str(colors[idx])

# labelme_jsons_train = glob(osp.join(json_dir, "train/*.json"))
# labelme_jsons_val = glob(osp.join(json_dir, "val/*.json"))

labelme_jsons_val = glob(osp.join(json_dir, "test/*.json"))
image_group_ele = SubElement(project_ele, 'ImageGroup')
# SubElement(image_group_ele, 'NumberOfImages').text = str(len(labelme_jsons_train) + len(labelme_jsons_val))
SubElement(image_group_ele, 'NumberOfImages').text = str(len(labelme_jsons_val))

count = 0
# for idx, labelme_jsons in tqdm(enumerate([(labelme_jsons_train, 'train'), (labelme_jsons_val, 'val')])):
for idx, labelme_jsons in tqdm(enumerate([(labelme_jsons_val, 'val')])):
    mode = labelme_jsons[1]
    for labelme_json in tqdm(labelme_jsons[0], desc=mode):
        filename = osp.split(osp.splitext(labelme_json)[0])[-1]
        
        # if filename != '3103':
        #     continue
        
        with open(labelme_json, 'r') as jf:
            labelme_anns = json.load(jf)
            
        width, height = labelme_anns['imageWidth'], labelme_anns['imageHeight']
        image_ele = SubElement(image_group_ele, 'Image')
        count += 1
        # SubElement(image_ele, 'Path').text = f'\\\\aiv1' + json_dir.replace("/", "\\") + f"\\{mode}\\{filename}.bmp"
        SubElement(image_ele, 'Path').text = f'\\\\aiv1' + json_dir.replace("/", "\\") + f"\\test\\{filename}.bmp"
        # SubElement(image_ele, 'Path').text = f'{filename}.bmp'
        SubElement(image_ele, 'Width').text = str(width)
        SubElement(image_ele, 'Height').text = str(height)
        SubElement(image_ele, 'SplitState').text = 'Training' if mode == 'train' else 'Validation'
        
        label_group_ele = SubElement(image_ele, 'LabelGroup')
        
        num_labels = 0
        is_normal = True
        for labelme_ann in labelme_anns['shapes']:
            label = labelme_ann['label']
            points = labelme_ann['points']
        
            if len(points) > 2:
                label_ele = SubElement(label_group_ele, 'Label')
                SubElement(label_ele, 'ClassIndex').text = str(class2index[label])
                SubElement(label_ele, 'Type').text = 'Contours'
                contour_group_ele = SubElement(label_ele, 'ContourGroup')
                contour_ele = SubElement(contour_group_ele, 'Contour', {'Type': 'Outer'})
                for point in points:
                    x = int(point[0])# - roi[0])
                    y = int(point[1])# - roi[1])
                    if x <= 0: x = 0
                    if y <= 0: y = 0
                    if x >= width: x = 1119
                    if y >= height: y = 767
                    
                    SubElement(contour_ele, 'Point', {'X': str(x), 'Y': str(y)})
                is_normal = False
                num_labels += 1
                
        if is_normal or num_labels == 0:
            SubElement(label_group_ele, 'IsNormal').text = 'true'
        else:
            SubElement(label_group_ele, 'IsNormal').text = 'false'
            
        SubElement(label_group_ele, 'NumberOfLabels').text = str(num_labels)
        
tree = ElementTree(project_ele)
tree.write(osp.join(output_dir), encoding='utf-8', xml_declaration=True)
            
    
            
# colors = ['rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
#           'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)']

# for label, color in zip(classes, colors[:len(classes)]):
#     neuro_anns['classes'].append({"name": label, "color": color})
                    
# with open(osp.join(output_dir, 'neuro.json'), 'w') as json_file:
#     json.dump(neuro_anns, json_file, ensure_ascii=False, indent=4)
    
    