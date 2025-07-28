
import inspect
from typing import List

import torch
from visionsuite.engines.utils.registry import OPTIMIZERS, Registry


def register_torch_optimizers() -> List[str]:
    TORCH_OPTIMIZERS = Registry("torch_optimizer", parent=OPTIMIZERS, scope="visionsuite.engines.utils.registry.optims")
    for module_name in dir(torch.optim):
        if module_name.startswith("__"):
            continue
        
        _object = getattr(torch.optim, module_name)
        if inspect.isclass(_object) and issubclass(_object, torch.optim.Optimizer):
            TORCH_OPTIMIZERS.register(module=_object)

    return TORCH_OPTIMIZERS


TORCH_OPTIMIZERS = register_torch_optimizers()
