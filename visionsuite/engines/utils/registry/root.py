from .build_functions import build_from_cfg
from .registry import Registry


MODELS = Registry('model', build_func=build_from_cfg)
OPTIMIZERS = Registry('optimzer', build_func=build_from_cfg)
DATASETS = Registry('dataset', build_func=build_from_cfg)
DATALOADERS = Registry('dataloader', build_func=build_from_cfg)
LOSSES = Registry('loss', build_func=build_from_cfg)
SCHEDULERS = Registry('scheduler', build_func=build_from_cfg)
LOOPS = Registry('loop', build_func=build_from_cfg)
RUNNERS = Registry('runner', build_func=build_from_cfg)
ENGINES = Registry('engine', build_func=build_from_cfg)
FUNCTIONALS = Registry('functional', build_func=build_from_cfg)
PIPELINES = Registry('pipelines', build_func=build_from_cfg)
SAMPLERS = Registry('samplers', build_func=build_from_cfg)
TRAINERS = Registry('trainers', build_func=build_from_cfg)
VALIDATORS = Registry('validators', build_func=build_from_cfg)
TESTERS = Registry('testers', build_func=build_from_cfg)

__all__ = [
    'build_from_cfg', 'Registry', 'MODELS', 'OPTIMIZERS', 'DATASETS', 'DATALOADERS', 'LOSSES', 'SCHEDULERS', 'LOOPS',
    'RUNNERS', 'ENGINES', 'FUNCTIONALS', 'PIPELINES', 'SAMPLERS', 'TRAINERS', 'VALIDATORS',
    'TESTERS']
