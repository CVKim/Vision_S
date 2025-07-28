import torch 
import torch.nn as nn

from visionsuite.engines.segmentation.utils.registry import LOSSES
from visionsuite.engines.utils.helpers import get_params_from_obj






@LOSSES.register()
class LossWithAux:
    def __init__(self, **config):
        # TODO: loss_name ???
        loss_obj = LOSSES.get(config['loss_name'], case_sensitive=True)
        loss_params = get_params_from_obj(loss_obj)
        for key in loss_params.keys():
            if key in config:
                loss_params[key] = config[key]
            
        self.loss = loss_obj(**loss_params)

    def __call__(self, inputs, target):
        # losses = {}
        # if isinstance(inputs, torch.Tensor):
        #     losses['out'] = self.loss(inputs, target)
        # else:
        #     for name, x in inputs.items():
        #         losses[name] = self.loss(x, target)

        # if len(losses) == 1:
        #     return losses["out"]

        # return losses["out"] + 0.5 * losses["aux"]
    
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] + 0.5 * losses["aux"]