import os.path as osp
from abc import abstractmethod

from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule
from visionsuite.engines.utils.torch_utils.utils import load_ckpt

class BaseTrainLoop(BaseOOPModule):
    
    required_attributes = ['model', 'train_dataloader', 'lr_scheduler', 'loss', 'optimizer', 'dataset',
                           'loops', 'start_epoch', 'epochs']
    
    def __init__(self, name=None):
        super().__init__(name=name)
        self.args = None 
        
        self.model = None 
        self.train_dataloader = None 
        self.val_dataloader = None 
        self.lr_scheduler = None 
        self.loss = None 
        self.optimizer = None
        self.scaler = None 
        
        self.dataset = None
        self.archive = None
        
        self._start_epoch = None
        self._epochs = None
        self.trainer = None 
        self.validator = None
        
        self._loops = []
        
    @property 
    def loops(self):
        return self._loops 
    
    @property 
    def start_epoch(self):
        return self._start_epoch 
    
    @property 
    def epochs(self):
        return self._epochs
    
    def set_epoch_for_sampler(self, epoch):
        if self.args['distributed']['use'] and self.dataset is not None and hasattr('train_sampler', self.dataset) and self.dataset.train_sampler is not None:
            self.dataset.train_sampler.set_epoch(epoch)

    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(*args, **kwargs)
        self.run_callbacks('on_loop_build_start')

        self.model = _model
        self.dataset = _dataset
        self.archive = _archive
            
    @BaseOOPModule.track_status
    @abstractmethod
    def run(self):
        self.log_info(f"START loop", self.run.__name__, __class__.__name__)  
        self.run_callbacks('on_loop_run_start')
    
    def _set_resume(self):
        if self.args['resume']['use']:
            assert osp.exists(self.args['resume']['seed_model']), ValueError(f"There is no such checkpoint: {self.args['resume']['seed_model']}")
            ckpt = load_ckpt(self.args['resume']['seed_model'])
            self.log_info(f"LOADED ckpt: {ckpt.keys()}", self._set_resume.__name__, __class__.__name__)
            
            self.model.model_without_ddp.load_state_dict(ckpt['model'], strict=True)
            self.log_info(f"LOADED seed_model weights", self._set_resume.__name__, __class__.__name__)
            
            attributes = ['optimizer', 'lr_scheduler', 'epoch']
            if self.args['train']['amp']:
                attributes.append('scaler')
            
            for attribute in attributes:
                if self.args['resume'][attribute]:
                    if attribute in ckpt:
                        if attribute == 'epoch':
                            self._start_epoch = int(ckpt[attribute]) + 1
                            if self.args['train']['epochs'] - 10 <= self.start_epoch <= self.args['train']['epochs'] + 10:
                                self._epochs = self.start_epoch + 100
                                self.log_info(f"Epochs is updated to {self.start_epoch + 100} since the loaded epoch is {int(ckpt[attribute])}", self._set_resume.__name__, __class__.__name__)
                        else:
                            getattr(self, attribute).load_state_dict(ckpt[attribute])
                    else:
                        raise AttributeError(f"There is no {attribute} in ckpt({self.args['train']['ckpt']})")
            self.log_info(f"SET resume lr_scheduler: start_epoch is {self._start_epoch}", self._set_resume.__name__, __class__.__name__) 
                    
        else:
            self._start_epoch = 1
            self._epochs = self.args['train']['epochs']
            self.log_info(f"NO resume lr_scheduler: start_epoch is {self._start_epoch}", self._set_resume.__name__, __class__.__name__)   
                