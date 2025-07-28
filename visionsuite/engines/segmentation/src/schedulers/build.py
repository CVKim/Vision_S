from visionsuite.engines.classification.utils.registry import SCHEDULERS
from visionsuite.engines.utils.helpers import get_params_from_obj


def build_scheduler(optimizer, epochs, iters_per_epoch, **config):
    
    warmup_total_iters = 0 
    if config['warmup_scheduler'] and 'total_iters' in config['warmup_scheduler'] and config['warmup_scheduler']['total_iters'] > 0:
        warmup_total_iters = config['warmup_scheduler']['total_iters']
    main_lr_scheduler = SCHEDULERS.get(config['type'], case_sensitive=config['case_sensitive'])(
        optimizer, 
        total_iters=iters_per_epoch * (epochs - warmup_total_iters), 
        power=config['power']
    )

    if config['warmup_scheduler'] and config['warmup_scheduler']['total_iters'] > 0:
        scheduler_obj = SCHEDULERS.get(config['warmup_scheduler']['type'], case_sensitive=config['warmup_scheduler']['case_sensitive'])
        if not scheduler_obj:
            raise RuntimeError(
                f"Invalid lr scheduler '{config['warmup_scheduler']['type']}'. Only ConstantLR and LinearLR are available")
        scheduler_params = get_params_from_obj(scheduler_obj)
        for key in scheduler_params.keys():
            if key in config['warmup_scheduler']:
                scheduler_params[key] = config['warmup_scheduler'][key]
                
            if key == 'optimizer':
                scheduler_params[key] = optimizer

        warmup_lr_scheduler = scheduler_obj(**scheduler_params)
        lr_scheduler = SCHEDULERS.get(config['warmup_scheduler']['integrate_scheduler']['type'],
                                      case_sensitive=config['warmup_scheduler']['integrate_scheduler']['case_sensitive'])(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config['warmup_scheduler']['total_iters']]
        )

    else:
        lr_scheduler = main_lr_scheduler
        
    return lr_scheduler