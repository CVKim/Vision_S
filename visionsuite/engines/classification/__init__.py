import visionsuite.engines.classification.src # to pre-compile
from visionsuite.engines.utils.bases import BaseEngine
from visionsuite.engines.classification.utils.registry import RUNNERS

class Engine(BaseEngine):
    def __init__(self, task):
        super().__init__(task=task)
        
    def train(self, cfgs_file=None, *args, **kwargs):
        
        # TODO: need to assign by yaml file
        runner = RUNNERS.get("TrainRunner", case_sensitive=True)('classification')
        assert runner is not None, ValueError(f"runner is None")
        
        runner.train(cfgs_file=cfgs_file, *args, **kwargs)
                        
        
        