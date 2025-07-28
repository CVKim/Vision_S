import os
import os.path as osp 
import yaml 
from typing import Union, Dict
from pathlib import Path 
import warnings

from visionsuite.utils.download import download_weights_from_url

def get_cfg(cfg: Union[str, Dict]):
    
    if isinstance(cfg, str):
        assert osp.exists(cfg), ValueError(f"There is no such cfg at {cfg}")
        with open(cfg) as yf:
            cfg = yaml.load(yf)
    assert isinstance(cfg, dict), ValueError(f"Parameters must be dict, not {type(cfg)} which is {cfg}")    
    print(f"* Successfully LOADED cfg: {cfg}")
    
    if 'output_dir' in cfg:
        if not osp.exists(cfg['output_dir']):
            os.mkdir(cfg['output_dir'])
        cfg['project'] = Path(cfg['output_dir'])
        del cfg['output_dir']
    else:
        warnings.warn(f"There is no output-dir assigned")
    return cfg


def get_params(params: Union[str, Dict]):
    if isinstance(params, str):
        assert osp.exists(params), ValueError(f"There is no such params at {params}")
        with open(params) as yf:
            params = yaml.load(yf)
    assert isinstance(params, dict), ValueError(f"Parameters must be dict, not {type(params)} which is {params}")    
    print(f"* Successfully LOADED params: {params}")

    return params


def get_weights(task: str, model_name: str, backbone: str,
                yolov10_url="https://huggingface.co/spaces/hamhanry/YOLOv10-OBB/resolve/main/pretrained/yolov10s-obb.pt",
                output_filename='/tmp/yolov10s-obb.pt'):
    weights = None
    
    if model_name == 'yolov8':
        if task == 'hbb_detection' or task == 'det':
            weights = f'{model_name}{backbone}.pt'
        elif task == 'obb_detection':
            weights = f'{model_name}{backbone}-obb.pt'
        elif task == 'instance_segmentation':
            weights = f'{model_name}{backbone}-seg.pt'
        else:
            NotImplementedError(f"There is no such weights for {model_name} and {backbone}")
    elif model_name == 'yolov10':
        if task == 'hbb_detection':
            weights = f'{model_name}{backbone}.pt'
        elif task == 'obb_detection':
            yolov10_url = yolov10_url.replace('yolov10s-obb.pt', f'yolov10{backbone}-obb.pt')
            output_filename = output_filename.replace('yolov10s-obb.pt', f'yolov10{backbone}-obb.pt')
            if download_weights_from_url(yolov10_url, output_filename):
                weights = output_filename
            else:
                raise RecursionError(f"Cannot download weights from {yolov10_url}")
        else:
            NotImplementedError(f"There is no such weights for {model_name} and {backbone}")
    elif model_name == 'yolov11':
        if task == 'hbb_detection' or task == 'det':
            weights = f'yolo11{backbone}.pt'
        elif task == 'obb_detection':
            weights = f'yolo11{backbone}-obb.pt'
        elif task == 'instance_segmentation':
            weights = f'yolo11{backbone}-seg.pt'
        else:
            NotImplementedError(f"There is no such weights for {model_name} and {backbone}")
    elif model_name == 'rtdetr':
        if task == 'hbb_detection':
            weights = f'{model_name}-{backbone}.pt'
        else:
            NotImplementedError(f"There is no such weights for {model_name} and {backbone}")
            
    assert weights is not None, RuntimeError(f"weights is wrong: {weights}")
    print(f"* Successfully DEFINED weights: {weights}")

    return weights