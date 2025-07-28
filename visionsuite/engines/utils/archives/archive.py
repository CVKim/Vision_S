import yaml
import os.path as osp
from visionsuite.engines.utils.helpers import create_output_dir, mkdir
from visionsuite.engines.utils.loggers.monitor import Monitor
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule


class Archive(BaseOOPModule):
    _train_dirs = ['val', 'cfgs', 'logs', 'weights']
    _test_dirs = ['cfgs', 'logs', 'vis']

    def __init__(self, mode, output_dir=None):
        super().__init__(__class__.__name__)
        self.args = None
        self._mode = mode 
        self._output_dir = output_dir
        
    @property 
    def mode(self):
        return self._mode 
            
    @property
    def output_dir(self):
        return self._output_dir
    
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self._output_dir = create_output_dir(self.args['output_dir'], mode=self._mode, make_dirs=True)
        assert osp.exists(self._output_dir), RuntimeError(f"Output dir ({self._output_dir}) has not been created")
        
        self.create_directories()
        self.set_logger(log_stream_level=self.args['logger']['log_stream_level'],
                        log_file_level=self.args['logger']['log_file_level'],
                        log_dir=self.logs_dir)        
        self._load()
        
    @BaseOOPModule.track_status
    def _load(self):
        self._load_monitor()
    
    @BaseOOPModule.track_status
    def _load_monitor(self):
        if self.args['monitor']['use']:
            self.monitor = Monitor()
            self.monitor.set(output_dir=self.logs_dir, fn='monitor')
            self.log_info(f"Monitor is LOADED and SET", self._load_monitor.__name__, __class__.__name__)

        else:
            self.monitor = None
            self.log_info(f"Monitor is NOT LOADED", self._load_monitor.__name__, __class__.__name__)
    
    @BaseOOPModule.track_status
    def save_args(self, args):
        with open(osp.join(self.cfgs_dir, 'args.yaml'), 'w') as yf:
            yaml.dump(args, yf, default_flow_style=False)
            
        self.log_info(f"Saved args at {osp.join(self.cfgs_dir, 'args.yaml')}", self.save_args.__name__, __class__.__name__)
    
    @BaseOOPModule.track_status
    def create_directories(self):
        for directory in getattr(self, f'_{self._mode}_dirs'):
            mkdir(osp.join(self._output_dir, directory))
            setattr(self, directory + '_dir', osp.join(self._output_dir, directory))    
    
            self.log_info(f"Created directory: {osp.join(self._output_dir, directory)}", self.create_directories.__name__, __class__.__name__)

        
        
            
    