from .base_runner import BaseRunner

            
class BaseTestRunner(BaseRunner):
    def test(self, cfgs_file=None, *args, **kwargs):
        
        self.set_configs(cfgs_file=cfgs_file, *args, **kwargs)
        self.set_variables()
        self.run()
    