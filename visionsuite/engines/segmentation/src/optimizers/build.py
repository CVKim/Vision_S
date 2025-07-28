from visionsuite.engines.segmentation.utils.registry import OPTIMIZERS
    

def build_optimizer(model, **config):
    
    from visionsuite.engines.utils.helpers import get_params_from_obj
    optim_obj = OPTIMIZERS.get(config['type'])
    optim_params = get_params_from_obj(optim_obj)
    for key in optim_params.keys():
        if key in config:
            optim_params[key] = config[key]
            
        if key == 'params':
            model.apply_aux_loss_to_params_to_optimize(config['lr'])
            optim_params[key] = model.params_to_optimize
        
    return optim_obj(**optim_params)
    
