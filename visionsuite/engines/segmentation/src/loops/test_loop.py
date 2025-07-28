from visionsuite.engines.utils.bases import BaseTestLoop
from visionsuite.engines.segmentation.utils.registry import LOOPS

@LOOPS.register()
class TestLoop(BaseTestLoop):
    
    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(_model=_model, _dataset=_dataset, _archive=_archive, *args, **kwargs)
        self.run_callbacks('on_loop_build_start')


        

        if 'dataloader' in self.args and self.args['dataloader'] is not None:
            from visionsuite.engines.segmentation.src.dataloaders.build import build_dataloader
            self.test_dataloader = build_dataloader(dataset=self.dataset, mode='test', 
                                                    **self.args['dataloader'], 
                                                    augment=self.args['augment'] if 'augment' in self.args else None)
            self.log_info(f"BUILT test_dataloader: {self.test_dataloader}", self.build.__name__, __class__.__name__)
            
        else:
            self.log_warning(f"NO dataloader", self.build.__name__, __class__.__name__)    

        if 'loss' in self.args and self.args['loss'] is not None:
            from visionsuite.engines.segmentation.src.losses.build import build_loss
            self.loss = build_loss(**self.args['loss'])
            self.log_info(f"BUILT loss: {self.loss}", self.build.__name__, __class__.__name__)  
        else:
            self.log_warning(f"NO loss", self.build.__name__, __class__.__name__)  
        
        from visionsuite.engines.segmentation.src.testers.build import build_tester
        tester = build_tester(**self.args['test']['tester'])()
        tester.build(model=self.model, loss=self.loss, dataloader=self.test_dataloader,
                            # label2index=self.dataset.label2index, 
                            archive=self.archive, **self.args['test'],
                            _logger=self.args['test'].get('logger', None))
        self.loops.append(tester)
        self.log_info(f"BUILT tester: {self.tester}", self.build.__name__, __class__.__name__)
            
        # self._set_resume()
                    
                    
    def run(self):
        for loop in self.loops:
            loop.run(0)