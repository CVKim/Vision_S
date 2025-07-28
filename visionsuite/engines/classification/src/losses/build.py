from visionsuite.engines.classification.utils.registry import LOSSES
from visionsuite.engines.utils.helpers import get_params_from_obj

def build_loss(**config):
    loss_obj = LOSSES.get(config['type'], case_sensitive=config['case_sensitive'])
    loss_params = get_params_from_obj(loss_obj)
    for key in loss_params.keys():
        if key in config:
            loss_params[key] = config[key]
        
    return loss_obj(**loss_params)
