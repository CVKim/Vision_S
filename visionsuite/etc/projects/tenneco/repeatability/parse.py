import os.path as osp 
import json
from glob import glob
from tqdm import tqdm
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math
 
 
def get_feret(points):
    poly = Polygon(points)
    min_rect = poly.minimum_rotated_rectangle

    if min_rect.geom_type == 'Polygon':
        rect_coords = list(min_rect.exterior.coords)[:-1] 
    elif min_rect.geom_type == 'LineString':
        rect_coords = list(min_rect.coords)
    else:
        rect_coords = []

    def distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    edges = []
    for i in range(len(rect_coords)):
        p1 = rect_coords[i]
        p2 = rect_coords[(i + 1) % len(rect_coords)]
        edges.append(distance(p1, p2))

    width = max(edges)
    height = min(edges)

    return width, height
 
def get_major_length(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_
    axes = pca.components_

    points_centered = points - center
    rotated = points_centered @ axes.T 

    width_major = rotated[:, 0].max() - rotated[:, 0].min()
    width_minor = rotated[:, 1].max() - rotated[:, 1].min()
    
    return width_major, width_minor
 
def polygon2rect(points):
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

def rect2polygon(points):
    return [[points[0][0], points[0][1]], [points[1][0], points[0][1]], [points[1][0], points[1][1]], [points[0][0], points[1][1]]]

def compute_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1

    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def get_merged_points(points1, points2, merge_threshold):
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)

    merge_threshold = 5
    distance = poly1.distance(poly2)

    if distance <= merge_threshold:
        buffered_union = unary_union([poly1.buffer(merge_threshold), poly2.buffer(merge_threshold)])
        merged = buffered_union.buffer(-merge_threshold)
        
        if merged.geom_type == 'MultiPolygon':
            merged = merged.convex_hull
    
        return True, merged
    else:
        return False, None
    
def merge(anns, config={'mark': 5}):
    
    new_anns = []
    candi_anns = {}
    for ann in anns:
        if ann['label'].lower() in config or ann['label'].upper() in config:
            if ann['label'].lower() in candi_anns or ann['label'].upper() in candi_anns:
                candi_anns[ann['label'].lower()].append(ann)
            else:
                candi_anns[ann['label'].lower()] = [ann]
        else:
            new_anns.append(ann)

    for key, val in candi_anns.items():
        dist = config[key]
        while True:
            init_length = len(val)
            delete_idxes, delete_jdxes = set(), set()
            for idx, candi_ann_1 in enumerate(val):
                for jdx, candi_ann_2 in enumerate(val[idx + 1:]):
                    jdx += idx + 1 
                    
                    if (candi_ann_1['shape_type'] == 'polygon' and candi_ann_2['shape_type'] == 'polygon') and (len(candi_ann_1['points']) < 3 or len(candi_ann_2['points']) < 3):
                        continue
                    elif candi_ann_1['shape_type'] == 'rectangle' and candi_ann_2['shape_type'] == 'rectangle':
                        candi_ann_1['points'] = rect2polygon(candi_ann_1['points'])
                        candi_ann_2['points'] = rect2polygon(candi_ann_2['points'])
                    else:
                        NotImplementedError
                    
                    is_merged, merged_points = get_merged_points(candi_ann_1['points'], candi_ann_2['points'], dist)
                    
                    if is_merged:
                        delete_idxes.add(idx)
                        delete_jdxes.add(jdx)
                        coordinates = list(merged_points.exterior.coords)
                        coord_list = [list(coord) for coord in coordinates]
                        candi_ann_1['points'] = coord_list
                        
                new_anns.append(candi_ann_1)
                break
        
            for delete_index in sorted(delete_idxes.union(delete_jdxes), reverse=True):
                del val[delete_index]
                
            if init_length == len(val):
                for _val in val:
                    new_anns.append(_val)
                
                break
                
def setbin(label, points, fov, specs={'mark': [{'fovs': [1, 2, 3, 10, 11, 12, 13, 14],
                                         'width': 52.6,
                                         'height': 35.1,
                                         'cond': 'and',
                                         },
                                        {'fovs': [4, 5, 6, 7, 8, 9],
                                         'width': 49.1,
                                         'height': 17.5,
                                         'cond': 'and',},
                                        {'fovs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                         'if': {
                                            # 'width longer than': 49.1,
                                            # 'height longer than': 17.5,
                                            'ratio': 0.25,
                                            'new label': 'scratch',
                                            'new width': 105.3,
                                            'new height': 14,
                                            'cond': 'and',
                                         }
                                        },
                                    ],
                                'chamfer_mark': [{'fovs': [1, 2, 3, 10, 11, 12, 13, 14],
                                         'width': 49.1,
                                         'height': 21.1,
                                         'cond': 'and',},
                                        {'fovs': [4, 5, 6, 7, 8, 9],
                                         'width': 49.1,
                                         'height': 17.5,
                                         'cond': 'and',}
                                    ],
                                'line': [{'fovs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                         'width': 200,
                                         'height': 14,
                                         'cond': 'and',},
                                    ]
                            }
    ):

    
    assert label.lower() in specs, ValueError(f"There is no such spce for {label.lower()}")
    spec = specs[label.lower()]
    
    for _spec in spec:
        if int(fov) not in _spec['fovs']:
            continue 
        
        is_ng, is_w_ng, is_h_ng = False, False, False
        x, y, _w, _h = polygon2rect(points)
        # w, h = get_major_length(points)
        w, h = get_feret(points)
        
        if 'if' in _spec:
            if 'ratio' in _spec['if'] and _spec['if']['ratio'] != 0:
                if min(w, h)/max(w, h) <= _spec['if']['ratio']:
                    w_th = _spec['if']['new width']
                    h_th = _spec['if']['new height']
                    cond = _spec['if']['cond']
                    label = _spec['if']['new label']
            elif 'width longer than' in _spec['if'] and 'height longer than' in _spec['if']:
                if w > _spec['if']['width longer than'] or h > _spec['if']['height longer than']:
                    w_th = _spec['if']['new width']
                    h_th = _spec['if']['new height']
                    cond = _spec['if']['cond']
                    label = _spec['if']['new label']
        else:
            w_th = _spec['width']
            h_th = _spec['height']
            cond = _spec['cond']
            label = label.lower()
            
        
        if w >= w_th:
            is_w_ng = True 
        
        if h >= h_th:
            is_h_ng = True 

        if cond == 'and':
            if is_w_ng and is_h_ng:
                is_ng = True 
            else:
                is_ng = False
        elif cond == 'or':
            if is_w_ng or is_h_ng:
                is_ng = True 
            else:
                is_ng = False 
            
    return is_ng, label
            
 
def run(base_dir, dir_name, case, roi,
    result_by_product, outputs_by_product):
    
    filenames_dir = osp.join(base_dir, dir_name, case, 'filenames')
    if not osp.exists(filenames_dir):
        os.mkdir(filenames_dir)
        
                                    
    json_files1 = sorted(glob(osp.join(base_dir, dir_name, f'{case}/exp/labels', '*.json')))
    for json_file1 in tqdm(json_files1, desc=f"{dir_name} > {case}: "):
        filename = osp.split(osp.splitext(json_file1)[0])[-1]
        
        # if '125020717193964' not in filename:
        #     continue
        
        json_file2 = osp.join(base_dir, dir_name, f'{case}/exp2/labels', f'{filename}.json')
        json_file3 = osp.join(base_dir, dir_name, f'{case}/exp3/labels', f'{filename}.json')
        assert osp.exists(json_file2), ValueError(f"There is no such file: {json_file2}")
        assert osp.exists(json_file3), ValueError(f"There is no such file: {json_file3}")
        
        with open(json_file1, 'r') as jf:
            anns1 = json.load(jf)['shapes']
        
        with open(json_file2, 'r') as jf:
            anns2 = json.load(jf)['shapes']
        
        with open(json_file3, 'r') as jf:
            anns3 = json.load(jf)['shapes']
            
        fov = int(filename.replace('_1_image', '').split("_")[-1])
        _filename = filename.replace('_1_image', '').split("_")[0]
        assert fov in range(1, 15), ValueError(f"fov {fov} must be in 1 ~ 14")
        if _filename not in result_by_product:
            outputs_by_product['count'] += 1
            result_by_product[_filename] = {'1': {}, '2': {}, '3': {}, 'case': case}


        def parse_by_product(result_by_product, anns, order):
            for ann in anns:
                label = ann['label']
                points = ann['points']
                
                if ann['shape_type'] == 'polygon' and len(points) < 3:
                    continue
                elif ann['shape_type'] == 'rectangle':
                    points = rect2polygon(points)

                new_points = []
                for point in points:
                    new_points.append([point[0] - roi[0], point[1] - roi[1]])

                if f'fov_{fov}' not in result_by_product[_filename][str(order)]:
                    is_ng, _label = setbin(label, new_points, fov)
                    result_by_product[_filename][str(order)][f'fov_{fov}'] = [{'class': _label,
                                                                    'points': new_points,
                                                                    'repeated': set(),
                                                                    'ng': is_ng}]
                else:
                    is_ng, _label = setbin(label, new_points, fov)
                    result_by_product[_filename][str(order)][f'fov_{fov}'].append({'class': _label,
                                                                    'points': new_points,
                                                                    'repeated': set(),
                                                                    'ng': is_ng})
        
        
        for order, anns in enumerate([anns1, anns2, anns3]):
            merge(anns)
            parse_by_product(result_by_product, anns, order + 1)
        
            
    for product_id, product_info in result_by_product.items():
        
        if 'repeated' in product_info:
            continue
        
        order_ngs = [False, False, False]
        for order_id, order_info in product_info.items():
            if order_id == 'case':
                continue
            for fov_id, fov_infos in order_info.items():
                if len(fov_infos) != 0:
                    for fov_info in fov_infos:
                        if fov_info['ng']: order_ngs[int(order_id) - 1] = True
                    
        if sum(order_ngs) == 3 or sum(order_ngs) == 0:
            product_info['repeated'] = True 
            outputs_by_product['repeated']['count'] += 1
            if sum(order_ngs) != 0:
                outputs_by_product['repeated']['ng_count'] += 1
                outputs_by_product['repeated']['ng_id'].append(product_id)
            else:
                outputs_by_product['repeated']['ok_count'] += 1
                outputs_by_product['repeated']['ok_id'].append(product_id)
        else:
            product_info['repeated'] = False
            outputs_by_product['not_repeated']['count'] += 1
            outputs_by_product['not_repeated']['id'].append(product_id)
                
    outputs_by_product['repeated_percentage'] = outputs_by_product['repeated']['count']/outputs_by_product['count']*100 if outputs_by_product['count'] != 0 else 0
                


if __name__ == '__main__':
    
    ##### 1st
    # case = '1st'
    # input_img_dir = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data'
    # base_dir = '/HDD/etc/repeatablility/talos3/1st/benchmark/'
    # dir_name = 'deeplabv3plus'
    # dir_name = 'deeplabv3plus_patch512'
    # dir_name = 'deeplabv3plus_w1120_h768'
    # dir_name = 'lps_w1120_h768'
    # dir_name = 'segnext_w1120_h768'
    
    # dir_name = 'mask2former_swin-s_w1120_h768'
    # dir_name = 'segformer_b2_unfrozen_w1120_h768'
    # dir_name = 'yolov12_xl'
    # dir_name = 'define'
    # dir_name = 'segformer_b2_unfrozen_w1120_h768_tta'
    # dir_name = 'segformer_b2_unfrozen_w1120_h768_nohsv_tta'
    # dir_name = 'segformer_b2_unfrozen_w512_h512_tta'
    # dir_name = 'define_sod'
    # dir_name = 'yolov12_xl_sod'
    # dir_name = 'segnext_w1120_h768_tta'
    # dir_name = 'neurocle'
    # defects = ['오염', '시인성', '딥러닝 바보', '한도 경계성', 'repeated_ok']

    # ###### 2nd
    # case = '2nd'
    # input_img_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/data'
    # base_dir = '/HDD/etc/repeatablility/talos3/2nd/benchmark/'
    # # dir_name = 'deeplabv3plus'
    # # dir_name = 'deeplabv3plus_patch512'
    # # dir_name = 'deeplabv3plus_w1120_h768'
    # # dir_name = 'mask2former_swin-s_w1120_h768'
    # # dir_name = 'lps_w1120_h768'
    # dir_name = 'segformer_b2_unfrozen_w1120_h768'
    # # dir_name = 'segformer_b2_unfrozen_w1120_h768_tta'
    # # dir_name = 'segformer_b2_unfrozen_w1120_h768_tta_v2'
    # # dir_name = 'yolov12_xl'
    # # dir_name = 'yolov12_xl_sod'
    # # dir_name = 'define'
    # # dir_name = 'define_sod'
    # # dir_name = 'segnext_w1120_h768'
    # # dir_name = 'segnext_w1120_h768_tta'
    # # dir_name = 'neurocle'
    # # defects = ['오염', '딥러닝', '경계성', 'repeated_ng', 'repeated_ok']
    # defects = ['오염', '시인성', '딥러닝 바보', '한도 경계성', '종횡비 경계성', '기타 불량', 'repeated_ok']

    ########################### etc.
    # case = '1st'
    # input_img_dir = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data'
    # base_dir = '/HDD/etc/repeatablility/talos2/1st/benchmark/'
    # dir_name = 'sage'
    # # dir_name = 'neurocle'
    # defects = ['오염', '딥러닝', '경계성', 'repeated_ng', 'repeated_ok']
    
    case = '2nd'
    input_img_dir = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data'
    base_dir = '/HDD/etc/repeatablility/talos2/2nd/benchmark/'
    dir_name = 'sage'
    # dir_name = 'neurocle'
    defects = ['오염', '딥러닝', '경계성', 'repeated_ng', 'repeated_ok']
    

    vis_repeated = False
    roi = [220, 60, 1340, 828]

    result_by_product = {}
    outputs_by_product = {'total_count': 0,
                        'repeated_count': 0,
                        'not_repeated_count': 0,
                        'repeated_percentage': 0}
    
    output_dir = osp.join(base_dir, dir_name, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    for defect in defects:
        output_by_product = {'count': 0,
                                'repeated': {
                                                'count': 0,
                                                'ng_count': 0,
                                                'ng_id': [], 
                                                'ok_count': 0,
                                                'ok_id': [],
                                },
                                'not_repeated': {
                                                'count': 0,
                                                'id': [],
                                }
                        }
        run(base_dir, dir_name, case=defect, roi=roi,
            result_by_product=result_by_product, outputs_by_product=output_by_product,
        )
        
        outputs_by_product[defect] = output_by_product
        outputs_by_product['total_count'] += output_by_product['count']
        outputs_by_product['repeated_count'] += output_by_product['repeated']['count']
        outputs_by_product['not_repeated_count'] += output_by_product['not_repeated']['count']
        outputs_by_product['repeated_percentage'] = outputs_by_product['repeated_count']/outputs_by_product['total_count']*100 if outputs_by_product['total_count'] != 0 else 0

    with open(osp.join(output_dir, 'outputs.json'), 'w') as jf:
        json.dump(outputs_by_product, jf, ensure_ascii=False, indent=4)
            
    class2label = {}
    width, height = 1760, 832
    overlaps = [0, 238, 197, 252, 249, 243, 265, 268, 217, 229, 257, 141, 157, 295]
    overlap = 1120

    offset_x = 250
    margin_x = 50
    margin_y = 50

    stitch = 2 # 1 or 2
    save_fov = False
    save_stitched = True
    add_weights = False
    # vis_repeated = False
    alpha = 0.5
    thickness = 2
    font_scale = 2
    iou_threshold = 0.05
    fov_indexes = [14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

            
    assert isinstance(overlap, int), ValueError(f"Overlap must be int, not {overlap} which is {type(overlap)}")
    for sample_id, orders_data in tqdm(result_by_product.items(), desc="ANALYZING: "):  
        
        # if sample_id != '125020717082139':
        #     continue
        
        for order, data in orders_data.items():
            if order == 'repeated':
                continue
            
            for fov_idx1, fov1 in enumerate(fov_indexes):
                if f'fov_{fov1}' in data:
                    defects1 = data[f'fov_{fov1}']
                else:
                    continue 
                
                for fov_idx2, fov2 in enumerate(fov_indexes[fov_idx1 + 1:]):
                    if f'fov_{fov2}' in data:
                        defects2 = data[f'fov_{fov2}']
                    else:
                        continue 
                
                    for defect1_idx, defect1 in enumerate(defects1):
                        _class1, points1, ng1  = defect1['class'], defect1['points'], defect1['ng']
                        for defect2_idx, defect2 in enumerate(defects2):
                            _class2, points2, ng2 = defect2['class'], defect2['points'], defect2['ng']
                                            
                            if _class1 == _class2:
                                
                                x1, y1, w1, h1 = polygon2rect(points1)
                                x2, y2, w2, h2 = polygon2rect(points2)
                                
                                iou = compute_iou(x1 - offset_x - margin_x/2, y1 + margin_y/2, w1  + margin_x, h1 + margin_y, 
                                                x2, y2, w2, h2)
                                                            
                                if iou > iou_threshold:
                                    data[f'fov_{fov1}'][defect1_idx]['repeated'].update([fov1, fov2])
                                    data[f'fov_{fov2}'][defect2_idx]['repeated'].update([fov1, fov2])
                            
    sample_cnt = 0
    for sample_id, orders_data in tqdm(result_by_product.items(), desc="VISUALIZAING: "):  
        sample_cnt += 1

        # if sample_id == '125032816410911':
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>....", orders_data['repeated'])
        # else:
        #     continue
        
        if orders_data['repeated'] and not vis_repeated:
            continue
        
        stitched_images = []
        vis_dir = osp.join(output_dir, 'repeated' if orders_data['repeated'] else 'not_repeated', 'vis', orders_data['case'])
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        for order, fovs_data in orders_data.items():
            if order in ['repeated', 'case']:
                continue
                            
            if save_fov:
                vis_fov_dir = osp.join(vis_dir, sample_id + f'_{order}')
                if not osp.exists(vis_fov_dir):
                    os.mkdir(vis_fov_dir)
            
            stitched_defects = []
            for fov_idx, fov in enumerate(fov_indexes):
                if f'fov_{fov}' in fovs_data:
                    defects = fovs_data[f'fov_{fov}']
                else:
                    defects = [[]]

                ### vis
                if case == '1st':
                    img_file = osp.join(input_img_dir, f'OUTER_shot0{order}', f'{sample_id}_{fov}_Outer', '1_image.bmp')
                elif case == '2nd':
                    img_file = osp.join(input_img_dir, order, f'{sample_id}_{fov}', '1_image.bmp')
                assert osp.exists(img_file), ValueError(f'There is no such image file: {img_file}')

                _img = cv2.imread(img_file)
                _img = _img[roi[1]:roi[3], roi[0]:roi[2]]

                is_defected = False
                for defect in defects:
                    if defect == []:
                        continue
                    is_defected = True
                    _class, points, is_ng = defect['class'].lower(), defect['points'], defect['ng']
                    
                    
                    if is_ng:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                    
                    x, y, w, h = polygon2rect(points)
                    
                    stitched_defects.append([_class, x + sum(overlaps[:fov_idx + 1]), y, w, h])
                        
                    if _class not in class2label:
                        class2label[_class] = len(class2label)
                    # lx, sx = get_major_length(points)
                    lx, sx = get_feret(points)
                    cv2.putText(_img, f'{_class}', (int(x), int(y - 50)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.3, tuple(map(int, color)), 1)
                    cv2.putText(_img, f'lx{lx:.1f}_sx{sx:.1f}', (int(x), int(y - 35)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.3, tuple(map(int, color)), 1)
                    cv2.putText(_img, f'w{w:.1f}_h{h:.1f}', (int(x), int(y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.3, tuple(map(int, color)), 1)
                    cv2.putText(_img, f'{_class}', (int(x), int(y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, tuple(map(int, color)), 2)
                    
                    # cv2.polylines(_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    ### rotate the bbox by the left top point
                    cv2.polylines(_img, [np.array(points, dtype=np.int32)], isClosed=True, color=tuple(map(int, color)), thickness=thickness)
                        
                    # ### original
                    # cv2.rectangle(_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 224), 1)
                    
                    if len(defect['repeated']) != 0:
                        _x, _y, _w, _h = polygon2rect(points)
                        cv2.rectangle(_img, (int(_x) - 100, int(_y) - 100), (int(_x + _w + 100), int(_y + _h + 100)), 
                                            (0, 255, 0), 7)

                if save_fov:
                    cv2.imwrite(osp.join(vis_fov_dir, f'{fov}.bmp'), _img)
                    
                if fov_idx == 0:
                    if is_defected:
                        cv2.rectangle(_img, (0, 0), (_img[:, _img.shape[1] - overlap:].shape[1], _img[:, _img.shape[1] - overlap:].shape[0]), (255, 0, 255), 5)
                    img = _img 
                    continue

                if save_stitched:
                    if stitch == 1:
                        if add_weights:
                            img[:, sum(overlaps[:fov_idx + 1]):] = cv2.addWeighted(img[:, sum(overlaps[:fov_idx + 1]):], alpha, _img[:, :_img.shape[1] - overlaps[fov_idx]], 1 - alpha, 0)
                        img = np.hstack((img, _img[:, _img.shape[1] - overlaps[fov_idx]:]))
                    elif stitch == 2:
                        if is_defected:
                            cv2.rectangle(_img, (0, 0), (_img[:, _img.shape[1] - overlap:].shape[1], _img[:, _img.shape[1] - overlap:].shape[0]), (255, 0, 255), 5)
                        img = np.hstack((img, _img[:, _img.shape[1] - overlap:]))
                    else:
                        raise ValueError(f'There is no such stitch method: {stitch}')
                
            if save_stitched:
                # if stitch == 1:
                #     for stitched_defect in stitched_defects:
                #         _class, x, y, w, h = stitched_defect
                        
                #         cv2.putText(img, _class, (x, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 
                #                             1, tuple(map(int, color_map[class2label[_class]])), font_scale, 2)

                #         rotated_pts = cv2.transform(np.array([np.array([
                #                                                     [0, 0],
                #                                                     [w, 0],
                #                                                     [w, h],
                #                                                     [0, h]
                #                                                 ], dtype=np.float32)]), cv2.getRotationMatrix2D((0, 0), angle, 1.0))[0] + np.array([x, y], dtype=np.float32)
                #         cv2.polylines(_img, [rotated_pts.astype(np.int32)], isClosed=True, color=tuple(map(int, color_map[class2label[_class]])), thickness=thickness)
                #         # cv2.drawContours(_img, [np.intp(cv2.boxPoints(((x + max(w, h)/2, y + min(w, h)/2), (max(w, h), min(w, h)), -angle)))], 0, tuple(map(int, color_map[class2label[_class]])), thickness)
                #         # cv2.drawContours(img, [np.intp(cv2.boxPoints(((x + min(w, h)/2, y + max(w, h)/2), (w, h), -angle)))], 0, tuple(map(int, color_map[class2label[_class]])), thickness)
                #         # cv2.rectangle(img, (x, y), (x + w, y + h), tuple(map(int, color_map[class2label[_class]])), 2)
                
                stitched_images.append(img)
                
        if save_stitched:
            max_width = max([img.shape[1] for img in stitched_images])
            max_height = max([img.shape[0] for img in stitched_images])
            sep_height = 100
            image = np.zeros((max_height*3 + sep_height*2, max_width, 3))
            for zdx, stitched_image in enumerate(stitched_images):
                image[zdx*(max_height + sep_height):(zdx + 1)*max_height + zdx*sep_height, :stitched_image.shape[1]] = stitched_image
                
            cv2.imwrite(osp.join(vis_dir, sample_id + '.bmp'), image)
            
            
        # if sample_cnt > 30:
        #     break
