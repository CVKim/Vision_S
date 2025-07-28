
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmseg.customs.datasets import MaskDataset
from mmseg.registry import MODELS
import torch

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent


def build_model(model):
    import torch.nn as nn
    """Build model.

    If ``model`` is a dict, it will be used to build a nn.Module object.
    Else, if ``model`` is a nn.Module object it will be returned directly.

    An example of ``model``::

        model = dict(type='ResNet')

    Args:
        model (nn.Module or dict): A ``nn.Module`` object or a dict to
            build nn.Module object. If ``model`` is a nn.Module object,
            just returns itself.

    Note:
        The returned model must implement ``train_step``, ``test_step``
        if ``runner.train`` or ``runner.test`` will be called. If
        ``runner.val`` will be called or ``val_cfg`` is configured,
        model must implement `val_step`.

    Returns:
        nn.Module: Model build from ``model``.
    """
    if isinstance(model, nn.Module):
        return model
    elif isinstance(model, dict):
        model = MODELS.build(model)
        return model  # type: ignore
    else:
        raise TypeError('model should be a nn.Module object or dict, '
                        f'but got {model}')


def main():
    
    cfg = Config.fromfile(ROOT / 'configs/models/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-512x512.py')

    inputs = torch.rand((2, 3, 512, 512))

    model = build_model(cfg.model)

    model = model.to('cuda:1')
    inputs = inputs.to('cuda:1')    
    outputs = model(inputs)
    


if __name__ == '__main__':
    main()
