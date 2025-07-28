from visionsuite.engines.classification.utils.registry import OPTIMIZERS
from visionsuite.engines.utils.torch_utils.utils import set_weight_decay

    
def get_optim_parameters(model, **config):
    custom_keys_weight_decay = []
    if config['bias_weight_decay'] is not None:
        custom_keys_weight_decay.append(("bias", config['bias_weight_decay']))
    if config['transformer_embedding_decay'] is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, config['transformer_embedding_decay']))
            
    parameters = set_weight_decay(
        model,
        config['weight_decay'],
        norm_weight_decay=config['norm_weight_decay'],
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    return parameters


def build_optimizer(model, **config):
    
    from visionsuite.engines.utils.helpers import get_params_from_obj
    optim_obj = OPTIMIZERS.get(config['type'])
    optim_params = get_params_from_obj(optim_obj)
    for key in optim_params.keys():
        if key in config:
            optim_params[key] = config[key]
            
        if key == 'params':
            optim_params[key] = get_optim_parameters(model, **config)
        
    return optim_obj(**optim_params)
    
