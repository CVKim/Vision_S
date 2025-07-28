import glob 
import os.path as osp 
import cv2 
import json
import os
from tqdm import tqdm 
import numpy as np
from copy import deepcopy
from shapely import Polygon, MultiPolygon

from visionsuite.utils.dataset.formats.labelme.utils import add_labelme_element, init_labelme_json, get_mask_from_labelme

def intersected_polygon(window, points):
    xmin, ymin = window[0]
    xmax, ymax = window[1]

    # 사각형 Window를 Polygon으로 정의
    window_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    # Points를 Polygon으로 정의
    points_polygon = Polygon(points)

    # 교차 다각형 구하기 (intersection)
    if not window_polygon.is_valid:
        window_polygon = window_polygon.buffer(0)
    if not points_polygon.is_valid:
        points_polygon = points_polygon.buffer(0)
    geoms = window_polygon.intersection(points_polygon)
    
    if not geoms.is_empty:
        if geoms.geom_type == 'Polygon':
            return [[val1, val2] for val1, val2 in list(geoms.exterior.coords)]
        elif geoms.geom_type == 'Multipolygon':
            raise NotImplementedError
            return [list(x.exterior.coords) for x in geoms.geoms]
    else:
        return None


def intersection(boxA, boxB):
    # Box coordinates
    xmin1, ymin1 = boxA[0]
    xmax1, ymax1 = boxA[1]
    xmin2, ymin2 = boxB[0]
    xmax2, ymax2 = boxB[1]
    
    # Calculate the (x, y) coordinates of the intersection rectangle
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    
    # Check if there is an intersection
    if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
        return [[inter_xmin, inter_ymin], [inter_xmax, inter_ymax]]
    else:
        # No intersection
        return None


def min_max_normalize(image_array, min_val, max_val):
    normalized_array = (image_array - min_val) / (max_val - min_val)
    return np.clip(normalized_array, 0, 1)  # 값을 0과 1 사이로 클리핑


def labelme2patch(input_dir, output_dir, modes, rois, patch_width, patch_height, 
                    input_formats = ['bmp'],
                    output_format='bmp', patch_overlap_ratio = 0.2,
                    norm_val=None, vis=False, include_positive=True, classes_to_include=None):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
    if vis:
        vis_dir = osp.join(output_dir, 'vis')
        if not osp.exists(vis_dir):
            os.mkdir(vis_dir)
            
    dx = int((1. - patch_overlap_ratio) * patch_width)
    dy = int((1. - patch_overlap_ratio) * patch_height)
    class2label = {}
    for mode in modes:
        _output_dir = osp.join(output_dir, mode)
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
        
        img_files = []
        for input_format in input_formats:
            img_files += glob.glob(osp.join(input_dir, mode, f'*.{input_format}'))
        for img_file in tqdm(img_files):
            filename = osp.split(osp.splitext(img_file)[0])[-1]
            json_file = osp.splitext(img_file)[0] + '.json'
            with open(json_file, 'r') as jf:
                anns = json.load(jf)['shapes']
            
            img = cv2.imread(img_file)
            img_h, img_w, img_c = img.shape
            
            if len(rois) == 1 and len(rois[0]) == 0:
                rois = [[0, 0, img_w, img_h]]
            
            num_patches = 0
            for roi in rois:
                for y0 in range(roi[1], roi[3], dy):
                    for x0 in range(roi[0], roi[2], dx):
                        if y0 + patch_height > roi[3]:
                            y = roi[3] - patch_height
                        else:
                            y = y0

                        if x0 + patch_width > roi[2]:
                            x = roi[2] - patch_width
                        else:
                            x = x0
                
                        _labelme = init_labelme_json(filename + f'_{num_patches}.{output_format}', img_w, img_h)
                        xmin, xmax, ymin, ymax = x, x + patch_width, y, y + patch_height
                        window = [[xmin, ymin], [xmax, ymax]]

                        for ann in anns:
                            included = False
                            label = ann['label'].lower()
                            if classes_to_include:
                                if label not in classes_to_include:
                                    continue
                            
                            if label not in class2label:
                                class2label[label] = len(class2label) + 1
                            points = ann['points']
                            shape_type = ann['shape_type']
                                                        
                            if shape_type == 'point':
                                if not include_positive:
                                    continue
                                _points = []
                                for point in points:
                                    _points.append([point[0] - xmin, point[1] -ymin])
                                _labelme = add_labelme_element(_labelme, ann['shape_type'], ann['label'], _points)
                                included = True
                            
                            elif shape_type == 'rectangle':
                                intersected_points = intersection(window, points)
                                
                                if intersected_points:
                                    included = True 
                                    
                                    for intersected_point in intersected_points:
                                        intersected_point[0] -= xmin
                                        intersected_point[1] -= ymin
                                    
                                    _labelme = add_labelme_element(_labelme, ann['shape_type'], ann['label'], intersected_points)
                            elif shape_type == 'polygon':
                                if len(points) <= 2:
                                    if include_positive:
                                        _points = []
                                        for point in points:
                                            _points.append([point[0] - xmin, point[1] -ymin])
                                        _labelme = add_labelme_element(_labelme, ann['shape_type'], ann['label'], _points)
                                    continue
                                intersected_points = intersected_polygon(window, points)
                                if intersected_points:
                                    included = True 
                                    for intersected_point in intersected_points:
                                        intersected_point[0] -= xmin
                                        intersected_point[1] -= ymin
                                    _labelme = add_labelme_element(_labelme, ann['shape_type'], ann['label'], intersected_points)
                            
                                
                                
                        if included:
                            if norm_val is not None:
                                if norm_val['type'] == 'min_max':
                                    patch = min_max_normalize(deepcopy(img[ymin:ymax, xmin:xmax, :]), norm_val['min_val'], norm_val['max_val'])
                                    patch = (patch * 255).astype(np.uint8)
                            else:
                                patch = deepcopy(img[ymin:ymax, xmin:xmax, :])
                            cv2.imwrite(osp.join(_output_dir, filename + f'_{num_patches}.{output_format}'), patch)
                            with open(osp.join(_output_dir, filename + f'_{num_patches}.json'), 'w') as jf:
                                json.dump(_labelme, jf)
                                
                            if vis:
                                import imgviz
                                
                                mask = get_mask_from_labelme(osp.join(_output_dir, filename + f'_{num_patches}.json'),
                                                            patch_width, patch_height, class2label, format='opencv')
                                
                                vis_img = np.zeros((patch_height, patch_width*2, 3))
                                vis_img[:, :patch_width, :] = patch
                                color_map = imgviz.label_colormap(50)
                                mask = color_map[mask.astype(np.uint8)].astype(np.uint8)
                                vis_img[:, patch_width:, :] = mask 
                                
                                cv2.imwrite(osp.join(vis_dir, filename + f'_{num_patches}.png'), vis_img)
                                        
                            num_patches += 1


if __name__ == '__main__':
        
    input_dir = '/DeepLearning/research/data/unittests/unit_cost_test/split_interojo_dataset'
    output_dir = '/DeepLearning/research/data/unittests/unit_cost_test/neurocle/split_interojo_dataset'
    modes = ['train', 'val', 'test']
    classes_to_include = None


    patch_overlap_ratio = 0.2
    patch_width = 512
    patch_height = 512
    rois = [[]]
    vis = False
    include_positive = True
    input_formats = ['bmp']
    output_format = 'bmp'

    # norm_val = {'type': 'min_max', 'min_val': 44, 'max_val': 235}
    norm_val = None
        
    labelme2patch(input_dir, output_dir, modes, rois, patch_width, patch_height, input_formats, output_format=output_format,
                    norm_val=norm_val, vis=vis, include_positive=include_positive, classes_to_include=classes_to_include)
                            
                        
                    
                



                


