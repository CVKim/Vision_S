import time 
import os.path as osp
from visionsuite.engines.utils.helpers import mkdir

def on_validator_build_start(validator, *args, **kwargs):
    pass

def on_validator_build_end(validator, *args, **kwargs):
    
    for attribute_name in validator.required_attributes:
        assert hasattr(validator, attribute_name), ValueError(f'{attribute_name} must be assgined in validator class')
        assert getattr(validator, attribute_name) is not None, ValueError(f"{attribute_name} is None for validator")
    

def on_validator_epoch_start(validator, *args, **kwargs):
    pass

def on_validator_epoch_end(validator, *args, **kwargs):
    # validator.metric_logger.synchronize_between_processes()

    def _save_val():
        if kwargs['epoch']%validator.archive.args['val']['save_freq_epoch'] == 0 and osp.exists(validator.archive.val_dir):
            vis_dir = osp.join(validator.archive.val_dir, str(kwargs['epoch']))
            if not osp.exists(vis_dir):
                mkdir(vis_dir)
               

            from visionsuite.engines.segmentation.utils.vis.vis_val import save_validation
            from visionsuite.engines.utils.functionals import denormalize

            save_validation(validator.model, validator.device, validator.dataloader.dataset, len(validator.label2index), 
                                kwargs['epoch'], vis_dir, denormalize=denormalize, input_channel=3, \
                            image_channel_order='bgr', validation_image_idxes_list=[])  

    def _save_results():
        validator.results.epoch = int(kwargs['epoch'])
    #     validator.results.loss = float(round(validator.metric_logger.meters['loss'].global_avg, 4))
    #     validator.results.accuracy = float(round(validator.metric_logger.meters["acc1"].global_avg, 4))
        validator.results.time_for_a_epoch = float(round(time.time() - kwargs['start_time_epoch'], 3))

    _save_val()
    _save_results()

def on_validator_batch_start(validator, *args, **kwargs):
    pass

def on_validator_batch_end(validator, *args, **kwargs):
    pass 

def on_validator_step_start(validator, *args, **kwargs): # iteration for a batch
    pass

def on_validator_step_end(validator, *args, **kwargs): # iteration for a batch
    pass

callbacks = {
    "on_validator_build_start": [on_validator_build_start],
    "on_validator_build_end": [on_validator_build_end],
    "on_validator_epoch_start": [on_validator_epoch_start],
    "on_validator_epoch_end": [on_validator_epoch_end],
    "on_validator_batch_start": [on_validator_batch_start],
    "on_validator_batch_end": [on_validator_batch_end],
    "on_validator_step_start": [on_validator_step_start],
    "on_validator_step_end": [on_validator_step_end],
}
