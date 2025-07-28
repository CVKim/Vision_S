import time
import torch
import torch.nn as nn
from abc import abstractmethod

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.metrics.smoothed_value import SmoothedValue
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies
from visionsuite.engines.classification.utils.registry import TRAINERS
from visionsuite.engines.utils.system.gpu_logger import GPULogger
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.results import TrainResults
from .callbacks import callbacks

@TRAINERS.register()
class BaseTrainer(BaseOOPModule, Callbacks):
    
    required_attributes = ['model', 'loss', 'dataloader']
    
    def __init__(self):
        BaseOOPModule.__init__(self, name="BaseTrainer")
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
        self.model.model.train()
        self.metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        self.metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"
        start_time_epoch = time.time()
        self.log_info(f"START train epoch: {epoch}", self.run.__name__, __class__.__name__)
        for step, (image, target) in enumerate(self.metric_logger.log_every(self.dataloader, self.args['print_freq'], header)):
            self.run_callbacks('on_trainer_batch_start')
            start_time = time.time()
            image, target = image.to(self.args['device']), target.to(self.args['device'])
            self.log_debug(f"- image: {image.shape} with device({self.args['device']})", self.run.__name__, __class__.__name__)
            self.log_debug(f"- target: {target.shape} with device({self.args['device']})", self.run.__name__, __class__.__name__)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model.model(image)
                loss = self.loss(output, target)
                self.log_debug(f"- loss: {loss}", self.run.__name__, __class__.__name__)
                
            self._backward(loss)
            self._reset_ema_buffer(epoch, step)
            self._update_logger(output, target, loss, start_time, batch_size = image.shape[0])

            self.run_callbacks('on_trainer_batch_end')

        self.lr_scheduler.step()

        self.run_callbacks('on_trainer_epoch_end', 
                           epoch=epoch, start_time_epoch=start_time_epoch)

    def _backward(self, loss):
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.log_debug(f"- Backward with scaler({self.scaler})", self._backward.__name__, __class__.__name__)
            if self.args['clip_grad_norm'] is not None:
                self.scaler.unscale_(self.optimizer)
                self.log_debug(f"- Unscaled the gradients of optimizer's assigned params", self._backward.__name__, __class__.__name__)
                nn.utils.clip_grad_norm_(self.model.model.parameters(), self.args['clip_grad_norm'])
                self.log_debug(f"- clip_grad_norm({self.args['clip_grad_norm']})", self._backward.__name__, __class__.__name__)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.log_debug(f"- Updated gradients", self._backward.__name__, __class__.__name__)

        else:
            loss.backward()
            self.log_debug(f"- Backward without scaler", self._backward.__name__, __class__.__name__)
            if self.args['clip_grad_norm'] is not None:
                nn.utils.clip_grad_norm_(self.model.model.parameters(), self.args['clip_grad_norm'])
                self.log_debug(f"- clip_grad_norm({self.args['clip_grad_norm']})", self._backward.__name__, __class__.__name__)
            self.optimizer.step()
            
    def _reset_ema_buffer(self, epoch, step):
        if self.model.model_ema and step%self.model.args['ema']['steps'] == 0:
            self.model.model_ema.update_parameters(self.model.model)
            self.log_debug(f"- Updated EMA Model parameters at step({step})", self._reset_ema_buffer.__name__, __class__.__name__)
            if epoch < self.model.args['warmup_scheduler']['total_iters']:
                self.model.model_ema.n_averaged.fill_(0)
                self.log_debug(f"- Reset ema buffer to keep copying weights during warmup period", self._reset_ema_buffer.__name__, __class__.__name__)
                
    def _update_logger(self, output, target, loss, start_time, batch_size):
        if self.metric_logger is not None:
            acc1, acc5 = get_accuracies(output, target, topk=(1, self.args['topk']))
            self.metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            self.metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            self.metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            self.metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            self.metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
        
            self.log_debug(f"- Updated metric_logger", self._update_logger.__name__, __class__.__name__)
            
        if self.gpu_logger is not None:
            self.gpu_logger.update()
            
            self.log_debug(f"- Updated gpu_logger", self._update_logger.__name__, __class__.__name__)
        