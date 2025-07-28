import os.path as osp
from typing import Union, Dict
    
try:
    from ultralytics import YOLO, RTDETR, settings
except (ImportError, AssertionError):
    import subprocess
    subprocess.run(["pip", "install", "ultralytics"])
    from ultralytics import YOLO, RTDETR, settings
    
settings.update({'wandb': False})

from visionsuite.cores.roboflow.utils.parsing import get_cfg, get_params, get_weights

def train(task: str=None, model_name: str=None, backbone: str=None, 
          recipe_dir=None,
          data: str=None, cfg: Union[str, Dict]=None, params: Union[str, Dict]=None):

    recipes = {'data': None, 'cfg': None, 'params': None}
    if recipe_dir is not None:
        for key in recipes.keys():
            if key == 'params':
                recipes[key] = osp.join(recipe_dir, f'train.yaml')
            else:
                recipes[key] = osp.join(recipe_dir, f'{key}.yaml')
    else:
        recipes['data'] = data
        recipes['cfg'] = cfg
        recipes['params'] = params
    for key, val in recipes.items():
        if key == 'param':
            key = 'train'
        assert osp.exists(val), ValueError(f"ERROR: There is no such reicpe for {key} at {val}")
            
    cfg = get_cfg(recipes['cfg'])
    params = get_params(recipes['params'])
    
    if 'task' in params:
        task = params['task']
        del params['task']
    if 'model_name' in params:
        model_name = params['model_name']
        del params['model_name']
    if 'backbone' in params:
        backbone = params['backbone']
        del params['backbone']

    assert task is not None, ValueError(f"ERROR: task must be assigned, not {task}")    
    assert model_name is not None, ValueError(f"ERROR: model_name must be assigned, not {model_name}")    
    assert backbone is not None, ValueError(f"ERROR: backbone must be assigned, not {backbone}")    
    
    weights = get_weights(task, model_name, backbone)
    
    if 'yolo' in model_name:
        model = YOLO(weights)
    elif model_name == 'rtdetr':
        model = RTDETR(weights)
    else:
        raise NotImplementedError(f'There is no such model supported: {model_name}')
    print(f"* Successfully LOADED model: {model_name}")
    
    print(f">>> Start to train")
    model.train(data=recipes['data'], **cfg, **params)
    print(f"* FINISHED training --------------------------------------")
    
if __name__ == '__main__':
    task = None
    model_name = None
    backbone = None
    
    # task = 'hbb_detection'
    # model_name = 'yolov8'
    # backbone = 'n'
    
    # recipe_dir = None
    # data = '/HDD/datasets/projects/visionsuite/yolo/hbb_detection/split_dataset_yolo_hbb/data.yaml'
    # cfg = '/HDD/datasets/projects/visionsuite/yolo/hbb_detection/split_dataset_yolo_hbb/cfg.yaml'
    # params = '/HDD/datasets/projects/visionsuite/yolo/hbb_detection/split_dataset_yolo_hbb/train.yaml'
    
    # recipe_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/recipes/v8'
    recipe_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/recipes/yolo'
    data = None
    cfg = None
    params = None
    
    train(task=task, model_name=model_name, backbone=backbone,
          recipe_dir=recipe_dir, 
          data=data, cfg=cfg, params=params)
