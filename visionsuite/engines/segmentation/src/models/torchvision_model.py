import torch
import torchvision 
import os.path as osp

from visionsuite.engines.segmentation.utils.registry import MODELS
from visionsuite.engines.utils.helpers import assert_key_dict
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule

        
@MODELS.register()
class TorchvisionModel(BaseOOPModule):
    """
    - weights can be referred by https://pytorch.org/vision/main/models.html
    """
    def __init__(self):
        super().__init__(name='TorchvisionModel')
        self.args = None
        self._device = 'cpu'
        self._model_without_ddp = None
        self._model_ema = None
        self._model = None 
        self._params_to_optimize = None
               
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        
        
        self._load_model() 
        self._to_device()
        self._set_dist()
        self._set_params_to_optimize()
        
        self._load_seed_model()
            
    @property
    def model(self):
        return self._model
    
    @property
    def model_without_ddp(self):
        return self._model_without_ddp
    
    @property
    def model_ema(self):
        return self._model_ema
    
    @property
    def params_to_optimize(self):
        return self._params_to_optimize
    
    @BaseOOPModule.track_status
    def _load_model(self):
        try:
            self._model = torchvision.models.get_model(
                                    self.args['model_name'],
                                    weights=self.args['weights'],
                                    weights_backbone=self.args['weights_backbone'],
                                    num_classes=self.args['num_classes'],
                                    aux_loss=self.args['aux_loss'],
                )
            
            self.log_info(f"LOADED model: {self.args['model_name']}", self._load_model.__name__, __class__.__name__)
        except Exception as error:
            raise RuntimeError(f"{error}: There has been error when loading torchvision model: {self.args['type']} with config({self.args}): ")
           
    @BaseOOPModule.track_status
    def _to_device(self):
        assert_key_dict(self.args['train'], 'device')
        self._device = self.args['train']['device']
        self._model.to(self._device)
        
        self.log_info(f"TRANSFER model to device: {self._device}", self._to_device.__name__, __class__.__name__)

    @BaseOOPModule.track_status
    def _set_dist(self):
        assert_key_dict(self.args, 'distributed')
        assert_key_dict(self.args['train'], 'sync_bn')

        if self.args['distributed'] and self.args['train']['sync_bn']:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            self.log_info(f"CONVERT batchnorm into sync_batchnorm: {self._device}", self._set_dist.__name__, __class__.__name__)
            
        self._model_without_ddp = self._model
        if self.args['distributed']:
            assert_key_dict(self.args['distributed'], 'gpu')
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self.args['distributed']['gpu']])
            self.log_info(f"SET distributed Data Parallel: {self.args['distributed']['gpu']}", self._set_dist.__name__, __class__.__name__)
            self._model_without_ddp = self.model.module
        else:
            assert_key_dict(self.args['train'], 'device_ids')
            if len(self.args['train']['device_ids']) > 1:
                self._model = torch.nn.parallel.DataParallel(self._model, device_ids=self.args['train']['device_ids'])
                self._model_without_ddp = self.model.module
                self.log_info(f"SET Data Parallel: {self.args['train']['device_ids']}", self._set_dist.__name__, __class__.__name__)
                
    def _set_params_to_optimize(self):
        self._params_to_optimize = [
            {"params": [p for p in self._model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in self._model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
        self.log_info(f"SET params to optimize", self._set_params_to_optimize.__name__, __class__.__name__)
        
    def apply_aux_loss_to_params_to_optimize(self, lr):
        if self.args['aux_loss']:
            params = [p for p in self._model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            self._params_to_optimize.append({"params": params, "lr": lr * 10})
            self.log_info(f"APPLY aux_loss to params to optimize: {self.args['aux_loss']}", self.apply_aux_loss_to_params_to_optimize.__name__, __class__.__name__)
            
    def _load_seed_model(self):
        if 'seed_model' in self.args and osp.exists(self.args['seed_model']):
            from visionsuite.engines.utils.torch_utils.utils import load_ckpt

            assert osp.exists(self.args['seed_model']), ValueError(f"There is no such checkpoint: {self.args['seed_model']}")
            ckpt = load_ckpt(self.args['seed_model'])
            self.log_info(f"LOADED ckpt: {ckpt.keys()}", self._load_seed_model.__name__, __class__.__name__)
            
            self._model_without_ddp.load_state_dict(ckpt['model'], strict=True)
            self._model.load_state_dict(ckpt['model'], strict=True)
            self.log_info(f"LOADED seed_model weights", self._load_seed_model.__name__, __class__.__name__)
                    
        else:
            self.log_info(f"NOT LOADED seed_model", self._load_seed_model.__name__, __class__.__name__)   
    