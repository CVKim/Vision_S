
import inspect
from typing import List

import torch
from visionsuite.engines.utils.registry import SCHEDULERS, Registry


def register_torch_schedulers() -> List[str]:
    TORCH_SCHEDULERS = Registry("torch_scheduler", parent=SCHEDULERS, scope="visionsuite.engines.utils.registry.schedulers")
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith("__"):
            continue
        
        _object = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_object):# and issubclass(_object, torch.optim.lr_scheduler._LRScheduler):
            TORCH_SCHEDULERS.register(module=_object)

    return TORCH_SCHEDULERS


TORCH_SCHEDULERS = register_torch_schedulers()

        