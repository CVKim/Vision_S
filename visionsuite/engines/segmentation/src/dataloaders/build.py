from visionsuite.engines.utils.torch_utils.utils import collate_fn
from visionsuite.engines.segmentation.utils.registry import DATALOADERS
from visionsuite.engines.utils.helpers import get_params_from_obj

def build_dataloader(dataset, mode, **config):
    try:
        dataloader_obj = DATALOADERS.get(config['type'], case_sensitive=config['case_sensitive'])
        dataloader_params = get_params_from_obj(dataloader_obj)
        for key in dataloader_params.keys():
            if key == 'dataset':
                dataloader_params[key] = getattr(dataset, f'{mode}_{key}', dataset)
                
            if key in config:
                dataloader_params[key] = config[key]
            elif key in config[mode]:
                dataloader_params[key] = config[mode][key]
                
                
        dataloader = dataloader_obj(**dataloader_params)

    except Exception as error:
        raise Exception(f"{error} at build_dataloader")
    
    return dataloader

