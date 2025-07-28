from visionsuite.engines.utils.bases import BaseTrainLoop


class Loop(BaseTrainLoop):
    
    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(_model=_model, _dataset=_dataset, _archive=_archive, *args, **kwargs)
        self.run_callbacks('on_loop_build_start')

        if 'dataloader' in self.args and self.args['dataloader'] is not None:
            from visionsuite.engines.classification.src.dataloaders.build import build_dataloader
            self.train_dataloader = build_dataloader(dataset=self.dataset, mode='train', 
                                                    **self.args['dataloader'], 
                                                    augment=self.args['augment'] if 'augment' in self.args else None)
            self.log_info(f"BUILT train_dataloader: {self.train_dataloader}", self.build.__name__, __class__.__name__)
            
            self.val_dataloader = build_dataloader(dataset=self.dataset, mode='val', 
                                                    **self.args['dataloader'], 
                                                    augment=self.args['augment'] if 'augment' in self.args else None)
            self.log_info(f"BUILT val_dataloader: {self.val_dataloader}", self.build.__name__, __class__.__name__)    
        else:
            self.log_warning(f"NO dataloader", self.build.__name__, __class__.__name__)    

        if 'loss' in self.args and self.args['loss'] is not None:
            from visionsuite.engines.classification.src.losses.build import build_loss
            self.loss = build_loss(**self.args['loss'])
            self.log_info(f"BUILT loss: {self.loss}", self.build.__name__, __class__.__name__)  
        else:
            self.log_warning(f"NO loss", self.build.__name__, __class__.__name__)  
        
        if 'optimizer' in self.args and self.args['optimizer'] is not None:
            from visionsuite.engines.classification.src.optimizers.build import build_optimizer
            self.optimizer = build_optimizer(model=self.model.model, **self.args['optimizer'])
            self.log_info(f"BUILT optimizer: {self.optimizer}", self.build.__name__, __class__.__name__)   
        else:
            self.log_warning(f"NO optimizer", self.build.__name__, __class__.__name__)   
        
        if 'scheduler' in self.args and self.args['scheduler'] is not None:
            from visionsuite.engines.classification.src.schedulers.build import build_scheduler
            self.lr_scheduler = build_scheduler(optimizer=self.optimizer, epochs=self.args['train']['epochs'], **self.args['scheduler'])
            self.log_info(f"BUILT lr_scheduler: {self.lr_scheduler}", self.build.__name__, __class__.__name__)     
        else:
            self.log_warning(f"NO lr_scheduler", self.build.__name__, __class__.__name__)     

        
        if 'train' in self.args and 'trainer' in self.args['train'] and self.args['train']['trainer'] is not None:
            from visionsuite.engines.classification.src.trainers.build import build_trainer
            trainer = build_trainer(**self.args['train']['trainer'])()
            trainer.build(model=self.model, loss=self.loss, optimizer=self.optimizer, 
                                lr_scheduler=self.lr_scheduler, dataloader=self.train_dataloader, 
                                scaler=self.scaler, archive=self.archive,
                                **self.args['train'],
                                _logger=self.args['trainer'].get('logger', None))
            self._loops.append(trainer)
            self.log_info(f"BUILT trainer: {self.trainer}", self.build.__name__, __class__.__name__)
        else:
            self.log_warning(f"NO trainer", self.build.__name__, __class__.__name__)
        
        if 'val' in self.args and 'validator' in self.args['val'] and self.args['val']['validator'] is not None:
            from visionsuite.engines.classification.src.validators.build import build_validator
            validator = build_validator(**self.args['val']['validator'])()
            validator.build(model=self.model, loss=self.loss, dataloader=self.val_dataloader,
                                label2index=self.dataset.label2index, 
                                device=self.args['train']['device'], topk=self.args['train']['topk'],
                                archive=self.archive, **self.args['val'],
                                _logger=self.args['validator'].get('logger', None))
            self._loops.append(validator)
            self.log_info(f"BUILT validator: {self.validator}", self.build.__name__, __class__.__name__)
        else:
            self.log_warning(f"NO validator", self.build.__name__, __class__.__name__)
            
        self._set_resume()
    
