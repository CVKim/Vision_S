from visionsuite.engines.segmentation.src.loops.build import build_loop
from visionsuite.engines.segmentation.src.datasets.build import build_dataset
from visionsuite.engines.segmentation.src.models.build import build_model
from visionsuite.engines.segmentation.utils.registry import RUNNERS
from visionsuite.engines.utils.bases import BaseTestRunner
from visionsuite.engines.utils.callbacks import Callbacks
from .callbacks import test_callbacks

@RUNNERS.register()
class TestRunner(BaseTestRunner, Callbacks):
    def __init__(self, task):
        BaseTestRunner.__init__(self, name='TestRunner', task=task)
        Callbacks.__init__(self)
        
        self.add_callbacks(test_callbacks)

    def set_configs(self, *args, **kwargs):
        super().set_configs(mode='test', *args, **kwargs)
        
        self.run_callbacks('on_runner_set_configs')
        
    def set_variables(self):
        super().set_variables()
        
        self.run_callbacks('on_runner_set_variables')
    
    def run(self):
        super().run()
        
        from visionsuite.engines.segmentation.src.datasets.mask_dataset import get_transform
        transform = get_transform(False, {"weights": None, "test_only": False, "backend": 'PIL', "use_v2": False})
        self.run_callbacks('on_runner_run_start')
        dataset = build_dataset(**self.args['test']['dataset'], transform=transform)
        self.log_info(f"Dataset is LOADED and BUILT", self.run.__name__, __class__.__name__)
        
        model = build_model(**self.args['model'])
        model.build(**self.args['model'], 
                    num_classes=len(self.args['test']['dataset']['classes']), 
                    train=self.args['test'], 
                    distributed=self.args['distributed']['use'],
                    _logger=self.args['model'].get('logger', None),
                    seed_model=self.args['test']['seed_model']
        )
        self.log_info(f"Model is LOADED and BUILT", self.run.__name__, __class__.__name__)
        
        loop = build_loop(**self.args['test']['loop'])
        loop.build(_model=model, 
                   _dataset=dataset, 
                   _archive=self._archive, 
                   **self.args,
                   _logger=self.args['loop'].get('logger', None)
                )
        self.log_info(f"Loop is LOADED and BUILT", self.run.__name__, __class__.__name__)

        loop.run()
        
        self.run_callbacks('on_runner_run_end')
