from visionsuite.engines.classification.utils.registry import SCHEDULERS


def build_scheduler(optimizer, epochs, **config):
    from visionsuite.engines.utils.helpers import get_params_from_obj
    if config['type'] == "CosineAnnealingLR":
        warmup_total_iters = 0 
        if config['warmup_scheduler'] and 'total_iters' in config['warmup_scheduler'] and config['warmup_scheduler']['total_iters'] > 0:
            warmup_total_iters = config['warmup_scheduler']['total_iters']
        main_lr_scheduler = SCHEDULERS.get(config['type'], case_sensitive=config['case_sensitive'])(
            optimizer, T_max = epochs - warmup_total_iters, eta_min=config['lr_min']
        )
    else:
        scheduler_obj = SCHEDULERS.get(config['type'], case_sensitive=config['case_sensitive'])
        if not scheduler_obj:
            raise RuntimeError(
                f"Invalid lr scheduler '{config['type']}'")
        scheduler_params = get_params_from_obj(scheduler_obj)
        for key in scheduler_params.keys():
            if key in config:
                scheduler_params[key] = config[key]
                
            if key == 'optimizer':
                scheduler_params[key] = optimizer
                
        main_lr_scheduler = scheduler_obj(**scheduler_params)
    
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

if __name__ == '__main__':
    import torch 
    import matplotlib.pyplot as plt

    # Example configuration for the scheduler
    config = {
        'type': 'StepLR',
        'case_sensitive': False,
        'eta_min': 0.001,
        'gamma': 0.1,
        'step_size': 30,
        'warmup_scheduler': {
            'type': 'LinearLR',
            'total_iters': 10,
            'start_factor': 0.1,
            'case_sensitive': False,
            'integrate_scheduler': {
                'type': 'SequentialLR',
                'case_sensitive': False
            }
        }
    }

    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD([torch.tensor(1.0)], lr=0.1)
    epochs = 100
    scheduler = build_scheduler(optimizer, epochs, config)

    # Track learning rate for each epoch
    lrs = []
    for epoch in range(epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), lrs, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()