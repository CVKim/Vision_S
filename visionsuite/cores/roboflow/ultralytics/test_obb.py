import os.path as osp
import os

from src.test_obb import test_obb

if __name__ == '__main__':
    
    model_name = 'yolov8'
    backbone = 'l_dfl_0.2_norad'
    weights_file = f"/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/outputs/dfl_0.2_norad/weights/best.pt"

    # input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/split_dataset_doosan/val'
    # json_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/split_dataset_doosan/val'
    # output_dir = f'/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/tests/doosan/{model_name}_{backbone}'
    input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/sfaw'
    json_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/sfaw'
    output_dir = f'/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/tests/sfaw/{model_name}_{backbone}'
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    compare_gt = True
    iou_threshold = 0.9
    conf_threshold = 0.1
    line_width = 3
    font_scale = 2
    imgsz = 768
    _classes = ['BOX']
    input_img_ext = 'bmp'
    output_img_ext = 'jpg'
    output_img_size_ratio = 1

    test_obb(weights_file, imgsz, _classes, input_dir, output_dir, json_dir, compare_gt, 
                iou_threshold, conf_threshold, line_width, font_scale, 
                input_img_ext=input_img_ext, output_img_ext=output_img_ext, output_img_size_ratio=output_img_size_ratio)