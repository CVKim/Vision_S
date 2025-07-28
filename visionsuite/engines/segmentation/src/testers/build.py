from visionsuite.engines.segmentation.utils.registry import TESTERS


def build_tester(**config):
    tester = TESTERS.get(config['type'], case_sensitive=config['case_sensitive'])
    
    return tester