import cv2
import json
import numpy as np
import os.path as osp

def get_text_coords(points, width, height, offset_w=0, offset_h=10):
    text_coord_x, text_coord_y = int(np.min(points, axis=0)[0]), int(np.min(points, axis=0)[1] - 10)
    if text_coord_x < offset_w:
        text_coord_x = int(np.max(points, axis=0)[0] + offset_w)
        
    if text_coord_x > width - 100:
        text_coord_x = 10
        
    if text_coord_y < offset_h:
        text_coord_y = int(np.max(points, axis=0)[1] + offset_h)
        
    if text_coord_y > height - 100:
        text_coord_y = 10
        
    return (text_coord_x, text_coord_y)

def get_filename(file_path, include_ext=False):
    if not include_ext:
        return osp.split(osp.splitext(file_path)[0])[-1]
    else:
        return osp.split(file_path)[-1]

def string_to_list_of_type(data, type_, lower=False, sep=','):

    if lower:
        return list(type_(value.strip().lower()) for value in data.split(sep)) if sep != "" else list(
            type_(value.strip().lower()) for value in list(data))
    else:
        return list(type_(value.strip())
                    for value in data.split(sep)) if sep != "" else list(type_(value.strip()) for value in list(data))

def letterbox(im, new_shape, color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, ratio, (dw, dh)

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)
        
        
# def check_corrupt_image():
    