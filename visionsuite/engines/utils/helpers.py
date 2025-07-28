import os 
import warnings
import datetime
import os.path as osp
from glob import glob 
from pathlib import Path
import re

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path



def print_class_name_on_instantiation(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        print(f"Instantiating class: {cls.__name__}")
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls

def mkdir(path, make_dirs=False):
    if not osp.exists(path):
        if make_dirs:
            os.makedirs(path)
        else:
            os.mkdir(path)
            
        print(f"Created directory: {path}")
    else:
        warnings.warn(f"There is already {path}")

def get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def create_output_dir(output_dir, mode, make_dirs=False):
    now = datetime.datetime.now()
    year = now.year 
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    
    output_dir = osp.join(output_dir, f'{year}_{month}_{day}_{hour}_{minute}_{second}', mode)
    mkdir(output_dir, make_dirs=make_dirs)
    
    return output_dir
    
def yaml2namespace(args_file):
    from types import SimpleNamespace
    import yaml 
    
    with open(args_file, 'r') as yf:
        cfgs = yaml.safe_load(yf)
        
    args = SimpleNamespace(**cfgs)
    
    return args
    
def yaml2dict(args_file):
    import yaml 
    
    with open(args_file, 'r') as yf:
        cfgs = yaml.safe_load(yf)
        
    return cfgs

def assert_key_dict(dictionary, key):
    assert key in dictionary, ValueError(f"There is no key({key})")
    
def get_params_from_obj(obj):
    import inspect

    signature = inspect.signature(obj)

    # Extract the parameters
    parameters = {param.name: param.default for param in signature.parameters.values()}
    
    return parameters

def update_dict(A, B):
    for key, value in B.items():
        # assert key in A, ValueError(f"There is no such key({key})")
        if isinstance(value, dict) and key in A and isinstance(A[key], dict):
            update_dict(A[key], value)
        else:
            A[key] = value