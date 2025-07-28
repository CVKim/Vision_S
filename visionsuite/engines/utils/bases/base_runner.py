from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

from abc import abstractmethod
import os.path as osp
import warnings 

from visionsuite.engines.utils.helpers import yaml2dict, update_dict
from visionsuite.engines.utils.torch_utils.utils import parse_device_ids, set_torch_deterministic, get_device
from visionsuite.engines.utils.torch_utils.dist import init_distributed_mode
from visionsuite.engines.utils.archives import Archive
from visionsuite.engines.utils.loggers import Logger

            
class BaseRunner(Logger):
    def __init__(self, task, name=None):
        super().__init__(name=name)
        self._task = task
        self._mode = None
        self.pipelines = {}
        
    @property 
    def task(self):
        return self._task 
    
    @property 
    def mode(self):
        return self._mode 
    
    @abstractmethod
    def set_configs(self, mode, cfgs_file=None, default_cfgs_file=None, *args, **kwargs):
        self._mode = mode
        if cfgs_file and osp.exists(cfgs_file):
            cfgs = yaml2dict(cfgs_file)
        else:
            cfgs = {}
            warnings.warn(f'There is no such cfgs file: {cfgs_file}')

        def _parse_args(args):
            # device_ids
            args[mode]['device_ids'] = parse_device_ids(args[mode]['device_ids'])
                        
        if default_cfgs_file is None:
            default_cfgs_file=ROOT.parents[1] / self._task / f'cfgs/default.yaml'
        default_cfgs = yaml2dict(default_cfgs_file)

        update_dict(default_cfgs, cfgs)  
        update_dict(default_cfgs, kwargs)          
        
        self.args = default_cfgs
        _parse_args(self.args)


    @abstractmethod
    def set_variables(self):
        # logger
        for key in ['archive']:
            if key not in self.args:
                self.args[key] = {'logger': {}}
            
            if 'logger' not in self.args[key]:
                self.args[key]['logger'] = self.args['logger']
            else:
                for key2, val2 in self.args['logger'].items():
                    if key2 not in self.args[key]['logger']:
                        self.args[key]['logger'][key2] = val2
        
        self._archive = Archive(self.mode)
        self._archive.build(**self.args['archive'])
        self._archive.save_args(self.args)
        
        # logger
        for key in ['runner', 'model', 'loop', 'trainer', 'validator', 'tester', 'dataset']:
            if key not in self.args:
                self.args[key] = {'logger': {}}
            
            if 'logger' not in self.args[key]:
                self.args[key]['logger'] = self.args['logger']
                self.args[key]['logger']['logs_dir'] = self._archive.logs_dir
            else:
                for key2, val2 in self.args['logger'].items():
                    if key2 not in self.args[key]['logger']:
                        self.args[key]['logger'][key2] = val2
                self.args[key]['logger']['logs_dir'] = self._archive.logs_dir
                
        self.set_logger(log_stream_level=self.args['runner']['logger']['log_stream_level'],
                        log_file_level=self.args['runner']['logger']['log_file_level'],
                        log_dir=self.args['runner']['logger']['logs_dir']) 
        
        
        self.log_info(f"Args: {self.args}", self.set_variables.__name__, __class__.__name__)
        
        init_distributed_mode(self.args, self._mode)
        self.log_info(f"Initialize distribution mode: {self.args['distributed']['use']}", self.set_variables.__name__, __class__.__name__)
        
        set_torch_deterministic(self.args[self._mode]['use_deterministic_algorithms'])
        self.log_info(f"Set torch deterministic: {self.args[self._mode]['use_deterministic_algorithms']}", self.set_variables.__name__, __class__.__name__)

        self.args[self._mode]['device'] = get_device(self.args[self._mode]['device'])
        self.log_info(f"Set devices: {self.args[self._mode]['device']}", self.set_variables.__name__, __class__.__name__)
        
        
    @abstractmethod
    def run(self):
        pass 
    
