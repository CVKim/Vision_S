import time
import torch
from abc import abstractmethod

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.metrics.smoothed_value import SmoothedValue
from visionsuite.engines.segmentation.utils.registry import TRAINERS
from visionsuite.engines.utils.system.gpu_logger import GPULogger
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.segmentation.utils.results import TrainResults
from .callbacks import callbacks

@TRAINERS.register()
class BaseTrainer(BaseOOPModule, Callbacks):
    
    required_attributes = ['model', 'loss', 'dataloader']
    
    def __init__(self, name='BaseTrainer'):
        BaseOOPModule.__init__(self, name=name)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)
        
    def build(self, model, loss, optimizer, lr_scheduler, dataloader, scaler=None, archive=None, **args):
        super().build(**args)
        self.run_callbacks('on_trainer_build_start')
        
        self.epoch = None
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.args = args
        self.scaler = scaler
        self.archive = archive
        
        self.results = TrainResults()
        self.log_info(f"SET TrainResults", self.build.__name__, __class__.__name__)
        
        self.metric_logger = MetricLogger(delimiter="  ")
        self.log_info(f"SET MetricLogger", self.build.__name__, __class__.__name__)
        
        self.gpu_logger = GPULogger(self.args['device_ids'])
        self.log_info(f"SET GPULogger with devices({self.args['device_ids']})", self.build.__name__, __class__.__name__)
        
        self.run_callbacks('on_trainer_build_end')

    @abstractmethod
    def run(self, epoch):
        self.run_callbacks('on_trainer_epoch_start')
        self.epoch = epoch 
        start_time_epoch = time.time()
        self.model.model.train()
        self.metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch: [{epoch}]"
        self.log_info(f"START train epoch: {epoch}", self.run.__name__, __class__.__name__)
        for batch in self.metric_logger.log_every(self.dataloader, self.args['print_freq'], header):
            self.run_callbacks('on_trainer_batch_start')

            start_time = time.time()
            image, target = batch[0].to(self.args['device']), batch[1].to(self.args['device'])
            self.log_debug(f"- image: {image.shape} ({image.dtype}) with device({self.args['device']})", self.run.__name__, __class__.__name__)
            self.log_debug(f"- target: {target.shape} ({target.dtype}) with device({self.args['device']})", self.run.__name__, __class__.__name__)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model.model(image) # 'out': (bs num_classes(including bg) h w)
                loss = self.loss(output, target)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            self._update_logger(output, target, loss, start_time, batch_size=image.shape[0])
            
            self.run_callbacks('on_trainer_batch_end')

        self.run_callbacks('on_trainer_epoch_end', 
                           epoch=epoch, start_time_epoch=start_time_epoch)
    
                
    def _update_logger(self, output, target, loss, start_time, batch_size):
        if self.metric_logger is not None:
            self.metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            self.metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            self.metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
        
            self.log_debug(f"- Updated metric_logger", self._update_logger.__name__, __class__.__name__)
        if self.gpu_logger is not None:
            self.gpu_logger.update()
        
            self.log_debug(f"- Updated gpu_logger", self._update_logger.__name__, __class__.__name__)