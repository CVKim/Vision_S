from visionsuite.engines.classification.utils.registry import DATALOADERS


def build_dataloader(dataset, mode, **config):
    try:
        # TODO: integrate the below
        mixup_cutmix = None
        if mode in config['augment'] and config['augment'][mode] and 'mixup_cutmix' in config['augment'][mode]:
            mixup_cutmix = config['augment'][mode]['mixup_cutmix'] 
                                                                                                
        dataloader = DATALOADERS.get(config['type'], case_sensitive=config['case_sensitive'])(dataset=getattr(dataset, f'{mode}_dataset'), 
                                                                                                sampler=getattr(dataset, f'{mode}_sampler'), 
                                                                                                batch_size=config[mode]['batch_size'], 
                                                                                                workers=config[mode]['workers'],
                                                                                                mixup_cutmix=mixup_cutmix)
    except Exception as error:
        raise Exception(f"{error} at build_dataloader")
    
    return dataloader