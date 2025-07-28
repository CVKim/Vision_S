import glob 
import os
import os.path as osp 
import json 
import numpy as np
from visionsuite.utils.dataset.converters.utils import xyxy2xywh
from tqdm import tqdm
import warnings


def labelme2yolo_hbb(input_dir, output_dir, copy_image=True, 
                     image_ext='bmp', image_width=None, image_height=None):

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
            
        folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

        class2idx = {}
        for folder in folders:
            _output_labels_dir = osp.join(output_dir, 'labels', folder)
            if not osp.exists(_output_labels_dir):
                os.makedirs(_output_labels_dir)

            json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
            print(f"There are {len(json_files)} json files")

            for json_file in tqdm(json_files, desc=folder):
                filename = osp.split(osp.splitext(json_file)[0])[-1]
                txt = open(osp.join(_output_labels_dir, filename + '.txt'), 'w')
                with open(json_file, 'r') as jf:
                    _anns = json.load(jf)
                    anns = _anns['shapes']
                    
                    if 'imageHeight' in _anns and 'imageWidth' in _anns:
                        image_height, image_width = _anns['imageHeight'], _anns['imageWidth']
                    
                if copy_image:
                    import cv2
                    from shutil import copyfile
                    img_file = osp.join(input_dir, folder, filename + f'.{image_ext}')
                    
                    assert osp.exists(img_file), ValueError(f"There is no such image: {img_file}")
                    
                    image = cv2.imread(img_file)
                    image_width = image.shape[1]
                    image_height = image.shape[0]
                    _output_image_dir = osp.join(output_dir, 'images', folder)
                    if not osp.exists(_output_image_dir):
                        os.makedirs(_output_image_dir)
                    copyfile(img_file, osp.join(_output_image_dir, filename + f'.{image_ext}'))
                    
                if len(anns) != 0:
                    for ann in anns:
                        shape_type = ann['shape_type']
                        label = ann['label']
                        if label not in class2idx.keys():
                            class2idx.update({label: len(class2idx)})
                        points = ann['points']
                        xyxy = []
                        
                        if shape_type == 'point':
                            warnings.warn(f"Positive point is included")
                        
                        elif shape_type == 'rectangle':
                            for point in points:
                                xyxy.append(point[0])
                                xyxy.append(point[1])

                            if len(xyxy) > 4:
                                raise RuntimeError(f"shape type is rectangle, but there are more than 4 points")

                        elif shape_type == 'polygon' or shape_type == 'Watershed':
                            xs, ys = [], []
                            for point in points:
                                xs.append(point[0])
                                ys.append(point[1])

                            xyxy.append(np.max(xs))
                            xyxy.append(np.max(ys))
                            xyxy.append(np.min(xs))
                            xyxy.append(np.min(ys))
                        else: 
                            print(f"NotImplemented shape: {shape_type} for {json_file}")
                            continue

                            
                        assert image_width is not None and image_height is not None, RuntimeError(f"Image width is {image_width} and image height is {image_height}")
                        xywh = xyxy2xywh([image_height, image_width], xyxy)
                        txt.write(str(class2idx[label]) + ' ')
                        for kdx in range(len(xywh)):
                            if kdx == len(xywh) -1:
                                txt.write(str(round(xywh[kdx], 3)))
                            else:
                                txt.write(str(round(xywh[kdx], 3)) + ' ')
                        txt.write('\n')
                
            txt.close()

        txt = open(osp.join(output_dir, 'classe2idx.txt'), 'w')
        for key, val in class2idx.items():
            txt.write(f'{key}: {val}\n')
        txt.close()
        
        txt = open(osp.join(output_dir, 'idx2class.txt'), 'w')
        for key, val in class2idx.items():
            txt.write(f'{val}: {key}\n')
        txt.close()
                
        print("** FINISED to convert labelme2yolo_hbb")
    else:
        print(f"****** There is already {output_dir}")        
            
def labelme2yolo_iseg(input_dir, output_dir, image_ext,
                                       copy_image=True, image_width=None, image_height=None):

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
        folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]


        class2idx = {}
        for folder in folders:
            _output_labels_dir = osp.join(output_dir, 'labels', folder)
            if not osp.exists(_output_labels_dir):
                os.makedirs(_output_labels_dir)

            json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
            print(f"There are {len(json_files)} json files")

            for json_file in tqdm(json_files, desc=folder):
                filename = osp.split(osp.splitext(json_file)[0])[-1]
                txt = open(osp.join(_output_labels_dir, filename + '.txt'), 'w')
                with open(json_file, 'r') as jf:
                    anns = json.load(jf)['shapes']
                    
                if copy_image:
                    import cv2
                    from shutil import copyfile
                    img_file = osp.join(input_dir, folder, filename + f'.{image_ext}')
                    assert osp.exists(img_file), ValueError(f"There is no such image: {img_file}")
                    
                    image = cv2.imread(img_file)
                    image_width = image.shape[1]
                    image_height = image.shape[0]
                    _output_image_dir = osp.join(output_dir, 'images', folder)
                    if not osp.exists(_output_image_dir):
                        os.makedirs(_output_image_dir)
                    copyfile(img_file, osp.join(_output_image_dir, filename + f'.{image_ext}'))
                    
                    
                assert image_width != None and image_height != None, ValueError(f"The input size (width, height) must be assigned")
                    
                if len(anns) != 0:
                    for ann in anns:
                        shape_type = ann['shape_type']
                        label = ann['label']
                        if label not in class2idx.keys():
                            class2idx.update({label: len(class2idx)})
                        points = ann['points']
                        if shape_type == 'point' or len(points) <= 2:
                            continue
                        assert len(points) >= 3, RuntimeError(f"The number of polygon points must be more than 3, not {len(points)} with {shape_type}")

                        txt.write(str(class2idx[label]) + ' ')
                        for idx, point in enumerate(points):
                            if idx == len(points) -1:
                                txt.write(f'{round(point[0]/image_width, 3)} {round(point[1]/image_height, 3)}')
                            else:
                                txt.write(f'{round(point[0]/image_width, 3)} {round(point[1]/image_height, 3)} ')
                        txt.write('\n')
                
            txt.close()

        txt = open(osp.join(output_dir, 'classes2idx.txt'), 'w')
        for key, val in class2idx.items():
            txt.write(f'{val}: {key}\n')
        txt.close()
        
        txt = open(osp.join(output_dir, 'idx2class.txt'), 'w')
        for key, val in class2idx.items():
            txt.write(f'{val}: {key}\n')
        txt.close()
        
        print("** FINISED to convert labelme2yolo_iseg")
    else:
        print(f"****** There is already {output_dir}")
    
def labelme2yolo_obb(input_dir: str, output_dir: str=None, copy_image=True, image_ext='bmp'):
    from visionsuite.utils.dataset.converters.labelme2dota import convert_labelme2dota
    from visionsuite.utils.dataset.converters.dota2yolo import convert_dota2yolo_obb

    _output_dir = input_dir + '_dota'
    
    if not osp.exists(_output_dir):
        convert_labelme2dota(input_dir, _output_dir, copy_image=copy_image, image_ext=image_ext)
        print("** FINISED to convert labelme2dota")
    else:
        print(f"****** There is already {_output_dir}")

    if not osp.exists(output_dir):
        convert_dota2yolo_obb(_output_dir, output_dir, copy_image=copy_image, image_ext=image_ext)
        print("** FINISED to convert dota2yolo_obb")
    else:
        print(f"****** There is already {output_dir}")