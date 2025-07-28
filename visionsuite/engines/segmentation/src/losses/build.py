from visionsuite.engines.segmentation.utils.registry import LOSSES

def build_loss(**config):
    loss = LOSSES.get(config['type'], case_sensitive=config['case_sensitive'])(**config)

    return loss

