import argparse 
from types import SimpleNamespace
from visionsuite.engines.utils.loggers import Logger

class BaseOOPModule(Logger):
    def __init__(self, name=None):
        super().__init__(name)
        self._status = {}
        
    def track_status(func):
        def wrapper(self, *args, **kwargs):
            # Update the status dictionary for the function being executed
            self._status[func.__name__] = True
            return func(self, *args, **kwargs)
        return wrapper

    @property 
    def status(self):
        return self._status 
    
    @track_status
    def build(self, *args, **kwargs):
        print(f"args: ", args)
        print(f"kwargs: ", kwargs)
        
        if '_logger' in kwargs and kwargs['_logger'] and 'logs_dir' in kwargs['_logger']:
            self.set_logger(log_stream_level=kwargs['_logger'].get('log_stream_level', 'DEBUG'),
                        log_file_level=kwargs['_logger'].get('log_file_level', 'DEBUG'),
                        log_dir=kwargs['_logger']['logs_dir']) 
            
        
        if isinstance(kwargs, (argparse.Namespace, SimpleNamespace)):
            self.args = dict(kwargs)
        elif isinstance(kwargs, dict):
            self.args = kwargs
        else:
            NotImplementedError(f"NOT Considered this case for args({args}) and kwargs({kwargs})")
        
        assert self.args is not None, RuntimeError(f"Args for dataset is None")
        self.log_info(f"Loaded args: {self.args}", self.build.__name__, __class__.__name__)