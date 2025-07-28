from visionsuite.engines.classification.utils.registry import DATASETS

def build_dataset(transform=None, **config):
    dataset = DATASETS.get(config['type'], case_sensitive=config['case_sensitive'])
    dataset(transform=transform)
    
    return dataset()