from visionsuite.engines.classification.utils.registry import LOOPS

def build_loop(**config):
    loop = LOOPS.get(config['type'], case_sensitive=config['case_sensitive'])()

    return loop