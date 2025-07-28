import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import cv2
import numpy as np
from typing import Optional
import numpy.typing as npt
import PIL
import math
from tqdm import tqdm
import warnings


try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

def shape_to_mask(img_shape: tuple[int, ...], 
                    points: list[list[float]],
                    shape_type: Optional[str] = None,
                    line_width: int = 10,
                    point_size: int = 5,
                    ignore_shape_types=['point', 'line', 'linestrip']) -> npt.NDArray[np.bool_]:
    
    mask = PIL.Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    
    if shape_type not in ignore_shape_types:
        if shape_type == "circle":
            assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == "rectangle":
            assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
            x1, y1 = min(xy[0][0], xy[1][0]), min(xy[0][1], xy[1][1])
            x2, y2 = max(xy[0][0], xy[1][0]), max(xy[0][1], xy[1][1])
            xy = [(x1, y1), (x2, y2)]
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == "line":
            assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "linestrip":
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "point":
            assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        elif shape_type in [None, "polygon"]:
            if len(xy) <= 2:
                warnings.warn(f'There are less than 2 points with polygon shape-type') 
            else:
                assert len(xy) > 2, "Polygon must have points more than 2"
                draw.polygon(xy=xy, outline=1, fill=1)
        else:
            raise ValueError(f"shape_type={shape_type!r} is not supported.")
        
    return np.array(mask, dtype=bool)


def labelme2coco(input_dir, output_dir, noviz, 
                 assert_image_path, modes, only_json=False, 
                 ignore_shape_types=['point', 'line', 'linestrip']):
    # if osp.exists(args.output_dir):
    #     print("Output directory already exists:", args.output_dir)
    #     sys.exit(1)
    for mode in modes:
        
        if not only_json:
            os.makedirs(osp.join(output_dir, mode), exist_ok=True)
        if not noviz:
            os.makedirs(osp.join(output_dir, mode, "Visualization"), exist_ok=True)
        print("Creating dataset:", output_dir)

        now = datetime.datetime.now()

        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            licenses=[
                dict(
                    url=None,
                    id=0,
                    name=None,
                )
            ],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        class_name_to_id = {}
        out_ann_file = osp.join(output_dir, f"{mode}.json")
        label_files = glob.glob(osp.join(input_dir, mode, "*.json"))
        for image_id, filename in tqdm(enumerate(label_files), desc=mode):
            with open(filename, 'r') as jf:
                label_file = json.load(jf)


            
            base = osp.splitext(osp.basename(filename))[0]
            if base == '3_5_124071615341242_12_Outer':
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            if assert_image_path:
                assert base == osp.splitext(label_file['imagePath'])[0]
            image_ext = osp.splitext(label_file['imagePath'])[-1]
            out_img_file = osp.join(output_dir, mode, base + image_ext)

            img_file = osp.join(input_dir, mode, base + image_ext)
            img = cv2.imread(img_file)
            # img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    # file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    file_name=osp.split(osp.splitext(out_img_file)[0])[-1] + osp.splitext(out_img_file)[-1],
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file['shapes']:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = shape_to_mask(img.shape[:2], points, shape_type, ignore_shape_types=ignore_shape_types)

        
                if label not in class_name_to_id:
                    class_id = len(class_name_to_id) + 1
                    class_name_to_id[label] = class_id
                    data["categories"].append(
                        dict(
                            supercategory=None,
                            id=class_id,
                            name=label,
                        )
                    )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type not in ignore_shape_types:
                    if shape_type == "rectangle":
                        (x1, y1), (x2, y2) = points
                        x1, x2 = sorted([x1, x2])
                        y1, y2 = sorted([y1, y2])
                        points = [x1, y1, x2, y1, x2, y2, x1, y2]
                    elif shape_type == "circle":
                        (x1, y1), (x2, y2) = points
                        r = np.linalg.norm([x2 - x1, y2 - y1])
                        # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                        # x: tolerance of the gap between the arc and the line segment
                        n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                        i = np.arange(n_points_circle)
                        x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                        y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                        points = np.stack((x, y), axis=1).flatten().tolist()
                    else:
                        points = np.asarray(points).flatten().tolist()
                else:
                    points = []

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            if not noviz:
                viz = img
                if masks:
                    labels, captions, masks = zip(
                        *[
                            (class_name_to_id[cnm], cnm, msk)
                            for (cnm, gid), msk in masks.items()
                            if cnm in class_name_to_id
                        ]
                    )
                    viz = imgviz.instances2rgb(
                        image=img,
                        labels=labels,
                        masks=masks,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
                out_viz_file = osp.join(output_dir, mode, "Visualization", base + image_ext)
                imgviz.io.imsave(out_viz_file, viz)

        with open(out_ann_file, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    # parser.add_argument("--input_dir", default='/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/split_dataset')
    # parser.add_argument("--output_dir", default='/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/split_coco_dataset')
    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset'
    output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_coco_dataset'
    noviz = False
    assert_image_path = False
    modes = ['train', 'val']
    only_json = False
    ignore_shape_types = ['point', 'line', 'linestrip']
    
    labelme2coco(input_dir, output_dir, noviz, assert_image_path, modes, only_json=only_json, 
                 ignore_shape_types=ignore_shape_types)