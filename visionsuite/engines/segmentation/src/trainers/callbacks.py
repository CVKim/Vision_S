import os.path as osp
import psutil
import time 

from visionsuite.engines.utils.torch_utils.utils import save_on_master

def on_trainer_build_start(trainer, *args, **kwargs):
    pass        
     
def on_trainer_build_end(trainer, *args, **kwargs):
    
    for attribute_name in trainer.required_attributes:
        assert hasattr(trainer, attribute_name), ValueError(f'{attribute_name} must be assgined in trainer class')
        assert getattr(trainer, attribute_name) is not None, ValueError(f"{attribute_name} is None for trainer")

def on_trainer_epoch_start(trainer, *args, **kwargs):
    pass 

def on_trainer_epoch_end(trainer, *args, **kwargs):
    def _save_model():
        if trainer.archive and trainer.archive.weights_dir:
            checkpoint = {
                "model": trainer.model.model_without_ddp.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "lr_scheduler": trainer.lr_scheduler.state_dict(),
                "epoch": kwargs['epoch'],
                "args": trainer.args,
            }
            if trainer.model.model_ema:
                checkpoint["model_ema"] = trainer.model.model.model_ema.state_dict()
            if trainer.scaler:
                checkpoint["scaler"] = trainer.scaler.state_dict()
            
            if trainer.archive.args['model']['save_last']:
                save_on_master(checkpoint, osp.join(trainer.archive.weights_dir, "last.pth"))    
                
            if kwargs['epoch']%trainer.archive.args['model']['save_freq_epoch'] == 0:
                save_on_master(checkpoint, osp.join(trainer.archive.weights_dir, f"model_{kwargs['epoch']}.pth"))
    
    def _save_monitor():
        if trainer.archive and trainer.archive.monitor:
            trainer.archive.monitor.log({"learning rate": trainer.metric_logger.meters['lr'].value})
            trainer.archive.monitor.log({"train avg loss": trainer.metric_logger.meters['loss'].global_avg})
            trainer.archive.monitor.save()
  
    def _save_results():
        trainer.results.epoch = int(kwargs['epoch'])
        trainer.results.loss = float(round(trainer.metric_logger.meters['loss'].global_avg, 4))
        trainer.results.learning_rate = float(round(trainer.metric_logger.meters['lr'].value, 4))
    #     trainer.results.accuracy = float(round(trainer.metric_logger.meters["acc1"].global_avg, 4))
        trainer.results.cpu_usage = float(round(psutil.virtual_memory().used / 1024 / 1024 / 1024, 4))
        trainer.results.gpu_usage = trainer.gpu_logger.mean()
        trainer.results.time_for_a_epoch = float(round(time.time() - kwargs['start_time_epoch'], 3))
          
    _save_model()
    _save_monitor()
    _save_results()
    
    
def on_trainer_batch_start(trainer, *args, **kwargs):
    pass

def on_trainer_batch_end(trainer, *args, **kwargs):
    pass

def on_trainer_step_start(trainer, *args, **kwargs): # iteration for a batch
    pass

def on_trainer_step_end(trainer, *args, **kwargs): # iteration for a batch
    pass

callbacks = {
    "on_trainer_build_start": [on_trainer_build_start], 
    "on_trainer_build_end": [on_trainer_build_end], 
    "on_trainer_epoch_start": [on_trainer_epoch_start],
    "on_trainer_epoch_end": [on_trainer_epoch_end],
    "on_trainer_batch_start": [on_trainer_batch_start],
    "on_trainer_batch_end": [on_trainer_batch_end],
    "on_trainer_step_start": [on_trainer_step_start],
    "on_trainer_step_end": [on_trainer_step_end],
}

