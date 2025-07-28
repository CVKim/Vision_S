from .base_runner import BaseRunner

            
class BaseTrainRunner(BaseRunner):
    def train(self, cfgs_file=None, *args, **kwargs):
        
        self.set_configs(cfgs_file=cfgs_file, *args, **kwargs)
        self.set_variables()
        self.run()
    