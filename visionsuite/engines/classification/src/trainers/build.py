from visionsuite.engines.classification.utils.registry import TRAINERS


def build_trainer(**config):
    trainer = TRAINERS.get(config['type'], case_sensitive=config['case_sensitive'])
    
    return trainer