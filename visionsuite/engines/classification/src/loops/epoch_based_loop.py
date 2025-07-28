
from visionsuite.engines.classification.utils.registry import LOOPS
from visionsuite.engines.classification.src.loops.loop import Loop
from visionsuite.engines.utils.callbacks import Callbacks
from .callbacks import callbacks

@LOOPS.register()
class EpochBasedLoop(Loop, Callbacks):
    def __init__(self, name="EpochBasedLoop"):
        Loop.__init__(self, name=name)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)

    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(_model, _dataset, _archive=_archive, *args, **kwargs)

        self.run_callbacks('on_loop_build_end')
        
    def run(self):
        super().run()

        for epoch in range(self.start_epoch, self.epochs):
            self.set_epoch_for_sampler(epoch)
            for loop in self.loops:
                loop.run(epoch)
            
        self.run_callbacks('on_loop_run_end')