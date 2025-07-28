from visionsuite.engines.classification.utils.registry import MODELS

def build_model(**config):
    model = MODELS.get(config['type'], 
                       case_sensitive=config['case_sensitive'])()
    
    return model
