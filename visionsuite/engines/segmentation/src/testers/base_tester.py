import time
import torch
from abc import abstractmethod
import os.path as osp
from pathlib import Path 
import os 

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.metrics.smoothed_value import SmoothedValue
from visionsuite.engines.segmentation.utils.registry import TESTERS
from visionsuite.engines.utils.system.gpu_logger import GPULogger
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.utils.helpers import increment_path
from visionsuite.engines.segmentation.utils.vis.vis_test import vis_by_batch
from visionsuite.engines.utils.functionals import denormalize
from .callbacks import callbacks


@TESTERS.register()
class BaseTester(BaseOOPModule, Callbacks):
    
    required_attributes = ['model', 'loss', 'dataloader']
    
    def __init__(self, name='BaseTester'):
        BaseOOPModule.__init__(self, name=name)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)
        
    def build(self, model, loss, dataloader, scaler=None, archive=None, **args):
        super().build(**args)
        self.run_callbacks('on_tester_build_start')
        if ('output_dir' in self.args and self.args['output_dir'] is None) or 'output_dir' not in self.args:
            output_dir = osp.join(osp.splitext(self.args['seed_model'])[0], '../../../test')
            
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = str(increment_path(Path(output_dir) / "exp", exist_ok=False, mkdir=True))
            self.args['output_dir'] = output_dir
            
        assert osp.exists(self.args['output_dir']), ValueError(f"There is no such output-dir: {self.args['output_dir']}")
        
        self.epoch = None
        
        self.model = model
        self.loss = loss
        self.dataloader = dataloader
        self.scaler = scaler
        self.archive = archive
        
        self.metric_logger = MetricLogger(delimiter="  ")
        self.log_info(f"SET MetricLogger", self.build.__name__, __class__.__name__)
        
        self.gpu_logger = GPULogger(self.args['device_ids'])
        self.log_info(f"SET GPULogger with devices({self.args['device_ids']})", self.build.__name__, __class__.__name__)
        
        self.run_callbacks('on_tester_build_end')

    @abstractmethod
    def run(self, epoch):
        self.run_callbacks('on_tester_epoch_start')
        start_time_epoch = time.time()
        self.model.model.eval()
        header = f"Epoch: [{epoch}]"
        self.log_info(f"START test epoch: {epoch}", self.run.__name__, __class__.__name__)
        
        for batch_idx, batch in enumerate(self.metric_logger.log_every(self.dataloader, self.args['print_freq'], header)):
            self.run_callbacks('on_tester_batch_start')
            start_time = time.time()
            image, target, fname = batch[0].to(self.args['device']), batch[1].to(self.args['device']), batch[2]
            self.log_debug(f"- image: {image.shape} with device({self.args['device']})", self.run.__name__, __class__.__name__)
            self.log_debug(f"- target: {target.shape} with device({self.args['device']})", self.run.__name__, __class__.__name__)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model.model(image) # ';out': (bs num_classes(including bg) h w)
                loss = self.loss(output, target)

            vis_by_batch(output['out'], target, image, fname, epoch, batch_idx, output_dir=self.args['output_dir'],
                         denormalize=denormalize)
                
            self._update_logger(output, target, loss, start_time, batch_size=image.shape[0])
            
            self.run_callbacks('on_tester_batch_end')

        self.run_callbacks('on_tester_epoch_end', 
                           epoch=epoch, start_time_epoch=start_time_epoch)
    
                
    def _update_logger(self, output, target, loss, start_time, batch_size):
        if self.metric_logger is not None:
            self.metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            self.metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
        
            self.log_debug(f"- Updated metric_logger", self._update_logger.__name__, __class__.__name__)
        if self.gpu_logger is not None:
            self.gpu_logger.update()
        
            self.log_debug(f"- Updated gpu_logger", self._update_logger.__name__, __class__.__name__)