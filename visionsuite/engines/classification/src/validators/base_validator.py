import torch
import warnings
from abc import abstractmethod

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies
from visionsuite.engines.utils.torch_utils.dist import reduce_across_processes
from visionsuite.engines.classification.utils.registry import VALIDATORS
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.results import ValResults
from .callbacks import callbacks


@VALIDATORS.register()
class BaseValidator(BaseOOPModule, Callbacks):
    
    required_attributes = ['model', 'dataloader']
    
    def __init__(self):
        BaseOOPModule.__init__(self, name="BaseValidator")
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)
        
    def build(self, model, loss, dataloader, device, label2index, archive=None, print_freq=100, topk=3, **args):
        super().build(**args)
        self.run_callbacks('on_validator_build_start')
        self.epoch = None
        self.model = model.model_ema if model.model_ema else model.model
        self.loss = loss
        self.dataloader = dataloader
        self.label2index = label2index
        self.args = args
        self.log_suffix = "EMA" if model.model_ema else ""
        self.archive = archive
        self.device = device
        self.print_freq = print_freq
        self.topk = topk
        
        self.results = ValResults()
        self.log_info(f"SET ValResults", self.build.__name__, __class__.__name__)
        
        self.metric_logger = MetricLogger(delimiter="  ")
        self.log_info(f"SET MetricLogger", self.build.__name__, __class__.__name__)
        
        self.run_callbacks('on_validator_build_end')
        
    @abstractmethod
    def run(self, epoch):
        self.epoch = epoch
        if epoch%self.args['epoch'] == 0:
            self.model.eval()
            header = f"Val.: {self.log_suffix}"
            num_processed_samples = 0
            start_time_epoch = 0
            self.run_callbacks('on_validator_epoch_start')
            self.log_info(f"START val epoch: {epoch}", self.run.__name__, __class__.__name__)
            with torch.inference_mode():
                for image, target in self.metric_logger.log_every(self.dataloader, self.print_freq, header):
                    self.run_callbacks('on_validator_batch_start')
                    image = image.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    self.log_debug(f"- image: {image.shape} with device({self.device})", self.run.__name__, __class__.__name__)
                    self.log_debug(f"- target: {target.shape} with device({self.device})", self.run.__name__, __class__.__name__)
                    output = self.model(image)

                    self._update_logger(output, target, batch_size=image.shape[0])
                    num_processed_samples += image.shape[0]
                    self.run_callbacks('on_validator_batch_start')
                    
            # gather the stats from all processes
            num_processed_samples = reduce_across_processes(num_processed_samples)
            if (
                hasattr(self.dataloader.dataset, "__len__")
                and len(self.dataloader.dataset) != num_processed_samples
                and torch.distributed.get_rank() == 0
            ):
                # See FIXME above
                warnings.warn(
                    f"It looks like the dataset has {len(self.dataloader.dataset)} samples, but {num_processed_samples} "
                    "samples were used for the validation, which might bias the results. "
                    "Try adjusting the batch size and / or the world size. "
                    "Setting the world size to 1 is always a safe bet."
                )
            # print(f"{header} Acc@1 {self.metric_logger.acc1.global_avg:.3f} Acc@5 {self.metric_logger.acc5.global_avg:.3f}")
            self.run_callbacks('on_validator_epoch_end', epoch=epoch, 
                               start_time_epoch=start_time_epoch)

    def _update_logger(self, output, target, batch_size):
        if self.metric_logger is not None:
            loss = self.loss(output, target)
            acc1, acc5 = get_accuracies(output, target, topk=(1, self.topk))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            self.metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            
            self.log_debug(f"- Updated metric_logger", self._update_logger.__name__, __class__.__name__)
            
            