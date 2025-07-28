import os 
import pandas as pd
import os.path as osp
import json
from tqdm import tqdm 
import cv2
import numpy as np
import imgviz
import math
import ast
from shapely.geometry import Polygon

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    else:
        return val
    
def safe_json_loads(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    else:
        return val
    
def obb2poly_le90(rboxes):
    x_ctr, y_ctr, width, height, angle = rboxes 
    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5
    rects = np.array([
                        [tl_x, br_x, br_x, tl_x],  # x 좌표
                        [tl_y, tl_y, br_y, br_y],  # y 좌표
                    ])  
    sin, cos = np.sin(angle), np.cos(angle)
    M = np.array([
                    [cos, -sin],
                    [sin,  cos]
                ])  
    polys = M @ rects  # (2, 4)

    # 중심 좌표 더하기
    polys[0, :] += x_ctr  # x 좌표에 중심 x 더함
    polys[1, :] += y_ctr  # y 좌표에 중심 y 더함

    # (8,) shape으로 변환: x1, y1, x2, y2, ..., x4, y4
    poly_flat = polys.T.reshape(-1)
    pts = poly_flat.reshape(4, 2).astype(np.int32)  # shape: (4, 2)
    pts = pts.reshape(-1, 1, 2)  # shape: (4, 1, 2)

    return pts

def ond_dim_points_to_polygon(points):
    # points를 (x, y) 형식의 튜플 리스트로 변환
    return [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

def get_polygon_iou(point1, point2):
    # 두 다각형을 Polygon 객체로 변환
    poly1 = Polygon(ond_dim_points_to_polygon(point1))
    poly2 = Polygon(ond_dim_points_to_polygon(point2))
    
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
        
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
        
    # 교집합 영역 계산
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    # IoU 계산
    iou = intersection_area / (union_area + 0.000001)
    
    return iou, poly1.area,poly2.area, intersection_area

def polygon2rect(points):
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

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

def cpp_round(x):
    return int(math.floor(x + 0.5)) if x > 0 else int(math.ceil(x - 0.5))

def parse(df, roi, output_dir):
    '''
        {
            product_id: {
                            '1':
                                {
                                'fov_1': [
                                            {'class': abc, 'x': 2, 'y': 3, 'width': 2, 'height': 3},
                                            {'class': abc, 'x': 2, 'y': 3, 'width': 2, 'height': 3},
                                            ...
                                        ],
                                'fov_2': { ... }, ..., 'fov_14': { ... },
                                }
                            '2':
                                {
                                'fov_1': [
                                            {'class': abc, 'x': 2, 'y': 3, 'width': 2, 'height': 3},
                                            {'class': abc, 'x': 2, 'y': 3, 'width': 2, 'height': 3},
                                            ...
                                        ],
                                'fov_2': { ... }, ..., 'fov_14': { ... },
                                },
                            '3':
                                {
                                'fov_1': [
                                            {'class': abc, 'x': 2, 'y': 3, 'width': 2, 'height': 3},
                                            {'class': abc, 'x': 2, 'y': 3, 'width': 2, 'height': 3},
                                            ...
                                        ],
                                'fov_2': { ... }, ..., 'fov_14': { ... },
                                },
                            'repeated': False,
                        },
            'product_id': { ... },
            }
        }
    '''
    '''
    data column:
        'type': 'vision'
        'fovs': [
                 {'fov': 1, 'defects': [ 
                                        {'id': ..., 'class': 'KEY', 'name': 'KEY', 'ng': False, 
                                         'width': 2.3, 'height': 3.3, 'widthPix': 224, 'heightPix': 224, 
                                         'angle': 0, 'confirm': None, 'area': 0.0231, 
                                         'positionInFov': {'x': 232.123, 'y': 213.1231}}
                                    ], ...},
                 {'fov': 2, 'defects': [], ...},
                 {'fov': 3, 'defects': [], ...},
                 ...
            ]
    '''
    '''
        output_data: {
                        'total_count': total number of products,
                        'repeated': {
                                        'count': total number of repeated products,
                                        'ng_count': ,
                                        'ng_id': [], 
                                        'ok_count': ,
                                        'ok_id: [],
                        }
                        'not_repeated': {
                                        'total_count': total number of not repeated products,
                        }
                        
        }
        

    '''
    
    parsed_data = {}
    output_data = {'total_count': 0, 
                   'repeated': {'count': 0, 'ng_count': 0, 'ng_id': [], 'ok_count': 0, 'ok_id': []}, 
                   'not_repeated': {'count': 0}}
    
    exception_txt = open(osp.join(output_dir, '../exception.txt'), 'w')
    
    for sample_id, group in tqdm(df.groupby('sampleId'), desc="PARSING excel: "):
        # if sample_id != 125032816575591:
        #     continue
        group = group.sort_values('inspectedAt')
        parsed_data[str(sample_id)] = {}
        order_ngs = []
        for order, (idx, row) in enumerate(group.iterrows()):
            order += 1
            order_ngs.append(row['ng'])
            parsed_data[str(sample_id)][str(order)] = {}

            for fov_data in row['data']['fovs']:
                parsed_data[str(sample_id)][str(order)][f"fov_{fov_data['fov']}"] = []
                for defect_data in fov_data['defects']:
                    
                    ###
                    if defect_data['ng']:
                        parsed_defect_data = {'class': defect_data['class'], 
                                              'x': float(defect_data['positionInFov']['x']) - roi[0],
                                              'y': float(defect_data['positionInFov']['y']) - roi[1],
                                              'width': float(defect_data['widthPix']), # longaxis
                                              'height': float(defect_data['heightPix']), # shortaxis
                                              'angle': float(defect_data['angle']),
                                              'repeated': set()
                                            }
                        parsed_data[str(sample_id)][str(order)][f"fov_{fov_data['fov']}"].append(parsed_defect_data)
                        
            assert len(parsed_data[str(sample_id)][str(order)]) == 14, RuntimeError(f"There must be 14 fovs for each sample-id({sample_id}), now {len(parsed_data[str(sample_id)][order])}")
            
        if len(parsed_data[str(sample_id)]) != 3:
            print(len(parsed_data[str(sample_id)]), '>>>> ', sample_id)
            exception_txt.write(str(sample_id))
            exception_txt.write('\n')
            del parsed_data[str(sample_id)]
            continue
        
        assert len(parsed_data[str(sample_id)]) == 3, RuntimeError(f"There must be 3 order for each sample-id({sample_id}), now {len(parsed_data[str(sample_id)])}")

        output_data['total_count'] += 1
        ###
        assert len(order_ngs) == 3, RuntimeError(f"The number of order_ngs must be 3, not {len(order_ngs)}")      
        if sum(order_ngs) == 0 or sum(order_ngs) == 3:
            parsed_data[str(sample_id)]['repeated'] = True
            output_data['repeated']['count'] += 1
            if sum(order_ngs) == 0:
                output_data['repeated']['ok_count'] += 1
                output_data['repeated']['ok_id'].append(sample_id)
            elif sum(order_ngs) == 3:
                output_data['repeated']['ng_count'] += 1
                output_data['repeated']['ng_id'].append(sample_id)
            else:
                raise Exception(f"There is no such case for order_ngs: {order_ngs}")
        else:
            parsed_data[str(sample_id)]['repeated'] = False
            output_data['not_repeated']['count'] += 1
        
    ###
    output_data['repeated_percentage'] = output_data['repeated']['count']/output_data['total_count']*100
            
    return parsed_data, output_data
        
csv_file = '/HDD/etc/repeatablility/talos2/inspection(2025-05-14).xlsx'

#### 1st
case = '1st'
input_img_dir = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data'
sheet_name = '250207'
output_dir = '/HDD/etc/repeatablility/talos2/1st/outputs'


# ### 2nd
# case = '2nd'
# input_img_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/data'
# sheet_name = '250328'
# output_dir = '/HDD/etc/repeatablility/talos2/2nd/outputs'


if not osp.exists(output_dir):
    os.makedirs(output_dir)

'''
data: sim result data
sampleId: product_id
ng: 0/1
inspectedAt: time
'''

color_map= np.concatenate([
                            np.array([
                                [255, 0, 255],
                                [254, 128, 0],
                                [23, 187, 76],
                                [237, 0, 178]
                            ]),
                            imgviz.label_colormap()
                        ], axis=0)
class2label = {}
roi = [220, 60, 1340, 828]
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
alpha = 0.5
vis_repeated = False
thickness = 2
font_scale = 2
iou_threshold = 0.05
fov_indexes = [14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

df = pd.read_excel(csv_file, sheet_name=sheet_name)
# df = df.drop(columns=['lineKey', 'isNoSampleId', '_id', 'clientInspectionId', 'groupId', 'equipmentKey' ,'productKey', 'metadata'])
# data.to_excel(osp.join(output_dir, '2nd_unit.xlsx'), index=False)
# df['data'] = df['data'].apply(safe_literal_eval)
df['data'] = df['data'].apply(json.loads)

parsed_data, output_data = parse(df, roi, output_dir=output_dir)
with open(osp.join(output_dir, 'output_data.json'), 'w') as jf:
    json.dump(output_data, jf, ensure_ascii=False, indent=4)


assert isinstance(overlap, int), ValueError(f"Overlap must be int, not {overlap} which is {type(overlap)}")
    
for sample_id, orders_data in tqdm(parsed_data.items(), desc="ANALYZING: "):  
    
    # if sample_id != '125020717095782':
    #     continue
    
    for order, data in orders_data.items():
        if order == 'repeated':
            continue
        
        for fov_idx1, fov1 in enumerate(fov_indexes):
            defects1 = data[f'fov_{fov1}']
            
            if len(defects1) == 0:
                continue 
            
            for fov_idx2, fov2 in enumerate(fov_indexes[fov_idx1 + 1:]):
                defects2 = data[f'fov_{fov2}']
                
                if len(defects2) == 0:
                    continue 
            
                for defect1_idx, defect1 in enumerate(defects1):
                    _class1, x1, y1, w1, h1, angle1 = defect1['class'], defect1['x'], defect1['y'], \
                                                    defect1['width'], defect1['height'], defect1['angle']
                    for defect2_idx, defect2 in enumerate(defects2):
                        _class2, x2, y2, w2, h2, angle2 = defect2['class'], defect2['x'], defect2['y'], \
                                                    defect2['width'], defect2['height'], defect2['angle']
                                        
                        if _class1 == _class2:
                            
                            rotated_pts1 = cv2.transform(np.array([np.array([
                                                                [0, 0],
                                                                [w1, 0],
                                                                [w1, h1],
                                                                [0, h1]
                                                            ], dtype=np.float32)]), 
                                                        cv2.getRotationMatrix2D((0, 0), angle1, 1.0))[0] + np.array([x1, y1], 
                                                        dtype=np.float32)
                            x1, y1, w1, h1 = polygon2rect(rotated_pts1)
                                                         
                            rotated_pts2 = cv2.transform(np.array([np.array([
                                                                [0, 0],
                                                                [w2, 0],
                                                                [w2, h2],
                                                                [0, h2]
                                                            ], dtype=np.float32)]), 
                                                        cv2.getRotationMatrix2D((0, 0), angle2, 1.0))[0] + np.array([x2, y2], 
                                                        dtype=np.float32)
                            x2, y2, w2, h2 = polygon2rect(rotated_pts2)
                            
                            iou = compute_iou(x1 - offset_x - margin_x/2, y1 + margin_y/2, w1  + margin_x, h1 + margin_y, 
                                              x2, y2, w2, h2)
                                                        
                            # iou = compute_iou(x1 - offset_x - margin_x/2, y1 + margin_y/2, w1  + margin_x, h1 + margin_y, 
                            #                   x2, y2, w2, h2)
                            
                            if iou > iou_threshold:
                                data[f'fov_{fov1}'][defect1_idx]['repeated'].update([fov1, fov2])
                                data[f'fov_{fov2}'][defect2_idx]['repeated'].update([fov1, fov2])
                        
                    


sample_cnt = 0
for sample_id, orders_data in tqdm(parsed_data.items(), desc="VISUALIZAING: "):  
    sample_cnt += 1

    # if sample_id == '125032816410911':
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>....", orders_data['repeated'])
    # else:
    #     continue
       
    if orders_data['repeated'] and not vis_repeated:
        continue
       
    stitched_images = []
    vis_dir = osp.join(output_dir, 'repeated' if orders_data['repeated'] else 'not_repeated', 'vis')
    if not osp.exists(vis_dir):
        os.makedirs(vis_dir)
    for order, fovs_data in orders_data.items():
        if order == 'repeated':
            continue
                        
        if save_fov:
            vis_fov_dir = osp.join(vis_dir, sample_id + f'_{order}')
            if not osp.exists(vis_fov_dir):
                os.mkdir(vis_fov_dir)
        
        stitched_defects = []
        for fov_idx, fov in enumerate(fov_indexes):

            defects = fovs_data[f'fov_{fov}']

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
                _class, x, y, w, h, angle = defect['class'].lower(), defect['x'], defect['y'], \
                                            defect['width'], defect['height'], float(defect['angle'])
                
                # pts = obb2poly_le90([x + w/2, y + h/2, w, h, -angle])
                
                stitched_defects.append([_class, x + sum(overlaps[:fov_idx + 1]), y, w, h])
                    
                if _class not in class2label:
                    class2label[_class] = len(class2label)
                
                cv2.putText(_img, f'w{w:.1f}_h{h:.1f}_a{angle:.1f}', (int(x), int(y - 25)), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, tuple(map(int, color_map[class2label[_class]])), 1)
                cv2.putText(_img, f'{_class}', (int(x), int(y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, tuple(map(int, color_map[class2label[_class]])), 2)
                
                # cv2.polylines(_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                ### rotate the bbox by the left top point
                rotated_pts = cv2.transform(np.array([np.array([
                                                                [0, 0],
                                                                [w, 0],
                                                                [w, h],
                                                                [0, h]
                                                            ], dtype=np.float32)]), cv2.getRotationMatrix2D((0, 0), angle, 1.0))[0] + np.array([x, y], dtype=np.float32)
                cv2.polylines(_img, [rotated_pts.astype(np.int32)], isClosed=True, color=tuple(map(int, color_map[class2label[_class]])), thickness=thickness)
                # else:
                #     box = cv2.boxPoints(((x + w/2, y + h/2), (w, h), -angle))
                #     box = np.floor(box + 0.5).astype(np.int32)
                #     cv2.drawContours(_img, [box], 0, tuple(map(int, color_map[class2label[_class]])), thickness)
                    
                # ### original
                # cv2.rectangle(_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 224), 1)
                
                if len(defect['repeated']) != 0:
                    _x, _y, _w, _h = polygon2rect(rotated_pts)
                    cv2.rectangle(_img, (int(_x) - 100, int(_y) - 100), (int(_x + _w + 100), int(_y + _h + 100)), 
                                        (0, 255, 0), 7)

            if save_fov:
                cv2.imwrite(osp.join(vis_fov_dir, f'{fov}.bmp'), _img)
                
            if fov_idx == 0:
                img = _img 
                continue

            if save_stitched:
                if stitch == 1:
                    if add_weights:
                        img[:, sum(overlaps[:fov_idx + 1]):] = cv2.addWeighted(img[:, sum(overlaps[:fov_idx + 1]):], alpha, _img[:, :_img.shape[1] - overlaps[fov_idx]], 1 - alpha, 0)
                    img = np.hstack((img, _img[:, _img.shape[1] - overlaps[fov_idx]:]))
                elif stitch == 2:
                    if is_defected:
                        cv2.rectangle(_img, (0, 0), (_img[:, _img.shape[1] - overlap:].shape[1], _img[:, _img.shape[1] - overlap:].shape[0]), (0, 0, 255), 15)
                    img = np.hstack((img, _img[:, _img.shape[1] - overlap:]))
                else:
                    raise ValueError(f'There is no such stitch method: {stitch}')
            
        if save_stitched:
            if stitch == 1:
                for stitched_defect in stitched_defects:
                    _class, x, y, w, h = stitched_defect
                    
                    cv2.putText(img, _class, (x, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, tuple(map(int, color_map[class2label[_class]])), font_scale, 2)

                    rotated_pts = cv2.transform(np.array([np.array([
                                                                [0, 0],
                                                                [w, 0],
                                                                [w, h],
                                                                [0, h]
                                                            ], dtype=np.float32)]), cv2.getRotationMatrix2D((0, 0), angle, 1.0))[0] + np.array([x, y], dtype=np.float32)
                    cv2.polylines(_img, [rotated_pts.astype(np.int32)], isClosed=True, color=tuple(map(int, color_map[class2label[_class]])), thickness=thickness)
                    # cv2.drawContours(_img, [np.intp(cv2.boxPoints(((x + max(w, h)/2, y + min(w, h)/2), (max(w, h), min(w, h)), -angle)))], 0, tuple(map(int, color_map[class2label[_class]])), thickness)
                    # cv2.drawContours(img, [np.intp(cv2.boxPoints(((x + min(w, h)/2, y + max(w, h)/2), (w, h), -angle)))], 0, tuple(map(int, color_map[class2label[_class]])), thickness)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), tuple(map(int, color_map[class2label[_class]])), 2)
            
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

