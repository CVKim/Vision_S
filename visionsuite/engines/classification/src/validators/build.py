from visionsuite.engines.classification.utils.registry import VALIDATORS

def build_validator(**val_config):
    validator = VALIDATORS.get(val_config['type'], case_sensitive=True)
    
    return validator