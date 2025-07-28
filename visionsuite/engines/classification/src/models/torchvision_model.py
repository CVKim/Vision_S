import torch
import torchvision 


from visionsuite.engines.utils.torch_utils.ema import ExponentialMovingAverage
from visionsuite.engines.classification.utils.registry import MODELS
from visionsuite.engines.utils.helpers import assert_key_dict
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule

        
@MODELS.register()
class TorchvisionModel(BaseOOPModule):
    """
    - weights can be referred by https://pytorch.org/vision/main/models.html
    """
    def __init__(self, name="TorchvisionModel"):
        super().__init__(name=name)
        self.args = None
        self._device = 'cpu'
        self._model_without_ddp = None
        self._model_ema = None
        self._model = None 
               
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        
        self.load_model() 
        self.to_device()
        self.set_dist()
        self.set_ema()
        
    @property
    def model(self):
        return self._model
    
    @property
    def model_without_ddp(self):
        return self._model_without_ddp
    
    @property
    def model_ema(self):
        return self._model_ema
    
    @BaseOOPModule.track_status
    def load_model(self):
        # TODO: NOT include top to specify weights to change num_classes
        try:
            if self.args['weights']:
                self._model = torchvision.models.get_model(name=self.args['model_name'] + self.args['backbone'], 
                                                       weights=self.args['weights'])
                self.log_info(f"LOADED weights: {self.args['weights']}", self.load_model.__name__, __class__.__name__)
                
                self._model.fc = torch.nn.Linear(self._model.fc.in_features, self.args['num_classes'])
                self.log_info(f"CHANGED fc for num_classes({self.args['num_classes']})", self.load_model.__name__, __class__.__name__)
            else:
                self._model = torchvision.models.get_model(name=self.args['model_name'] + self.args['backbone'], 
                                                       num_classes=self.args['num_classes'])
                
            self.log_info(f"LOADED model: {self.args['model_name']}", self.load_model.__name__, __class__.__name__)
        except Exception as error:
            raise RuntimeError(f"{error}: There has been error when loading torchvision model: {self.args['type']} with config({self.args}): ")
           
    @BaseOOPModule.track_status
    def to_device(self):
        assert_key_dict(self.args['train'], 'device')
        self._device = self.args['train']['device']
        self._model.to(self._device)

    @BaseOOPModule.track_status
    def set_dist(self):
        assert_key_dict(self.args, 'distributed')
        assert_key_dict(self.args['train'], 'sync_bn')

        if self.args['distributed'] and self.args['train']['sync_bn']:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            
        self._model_without_ddp = self._model
        if self.args['distributed']:
            assert_key_dict(self.args['distributed'], 'gpu')
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self.args['distributed']['gpu']])
            self.log_info(f"Distributed Data Parallel is set", self.set_dist.__name__, __class__.__name__)
            self._model_without_ddp = self.model.module
        else:
            assert_key_dict(self.args['train'], 'device_ids')
            if len(self.args['train']['device_ids']) > 1:
                self._model = torch.nn.parallel.DataParallel(self._model, device_ids=[self.args['train']['device_ids']])
                self._model_without_ddp = self.model.module
                self.log_info(f"Data Parallel is set", self.set_dist.__name__, __class__.__name__)
                
    @BaseOOPModule.track_status
    def set_ema(self):
        assert_key_dict(self.args, 'ema')
        self._model_ema = None
        if self.args['ema']['use']:
            assert_key_dict(self.args, 'world_size')
            assert_key_dict(self.args, 'batch_size')
            assert_key_dict(self.args, 'epochs')
                
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = self.args['distributed']['world_size'] * self.args['train']['batch_size'] * self.args['ema_config']['steps'] / self.args['train']['epochs']
            alpha = 1.0 - self.args['ema']['decay']
            alpha = min(1.0, alpha * adjust)
            self._model_ema = ExponentialMovingAverage(self._model_without_ddp, 
                                                 device=self.args['train']['device'], 
                                                 decay=1.0 - alpha)
            
            self.log_info(f"SET EMA", self.set_ema.__name__, __class__.__name__)