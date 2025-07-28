from visionsuite.engines.utils.registry.losses import TORCH_LOSSES
from visionsuite.engines.utils.registry.optims import TORCH_OPTIMIZERS
from visionsuite.engines.utils.registry.schedulers import TORCH_SCHEDULERS

from visionsuite.engines.utils.registry import RUNNERS as ROOT_RUNNERS
from visionsuite.engines.utils.registry import MODELS as ROOT_MODELS
from visionsuite.engines.utils.registry import DATASETS as ROOT_DATASETS
from visionsuite.engines.utils.registry import DATALOADERS as ROOT_DATALOADERS
from visionsuite.engines.utils.registry import LOOPS as ROOT_LOOPS
from visionsuite.engines.utils.registry import PIPELINES as ROOT_PIPELINES
from visionsuite.engines.utils.registry import SAMPLERS as ROOT_SAMPLERS
from visionsuite.engines.utils.registry import FUNCTIONALS as ROOT_FUNCTIONALS
from visionsuite.engines.utils.registry import TRAINERS as ROOT_TRAINERS
from visionsuite.engines.utils.registry import VALIDATORS as ROOT_VALIDATORS
from visionsuite.engines.utils.registry import TESTERS as ROOT_TESTERS

from visionsuite.engines.utils.registry import Registry


scope = 'visionsuite.engines.segmentation.src'
RUNNERS = Registry('runners', parent=ROOT_RUNNERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.runners']
        )

MODELS = Registry('models', parent=ROOT_MODELS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.models']
        )

OPTIMIZERS = Registry('optimizers', parent=TORCH_OPTIMIZERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.optimizers']
        )

DATASETS = Registry('datasets', parent=ROOT_DATASETS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.datasets']
        )

DATALOADERS = Registry('dataloaders', parent=ROOT_DATALOADERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.dataloaders']
        )

LOSSES = Registry('losses', parent=TORCH_LOSSES, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.losses']
        )

SCHEDULERS = Registry('schedulers', parent=TORCH_SCHEDULERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.schedulers']
        )

LOOPS = Registry('loops', parent=ROOT_LOOPS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.loops']
        )

PIPELINES = Registry('pipelines', parent=ROOT_PIPELINES, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.pipelines']
        )

SAMPLERS = Registry('samplers', parent=ROOT_SAMPLERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.samplers']
        )

FUNCTIONALS = Registry('functionals', parent=ROOT_FUNCTIONALS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.functionals']
        )

TRAINERS = Registry('trainers', parent=ROOT_TRAINERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.trainers']
        )

VALIDATORS = Registry('validators', parent=ROOT_VALIDATORS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.validators']
        )

TESTERS = Registry('testers', parent=ROOT_TESTERS, 
                   scope=scope,
                   locations=['visionsuite.engines.segmentation.src.testers']
        )
