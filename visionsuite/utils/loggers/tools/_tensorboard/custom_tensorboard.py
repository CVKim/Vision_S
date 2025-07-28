import tensorflow as tf
import datetime

class CustomTensorboard:
    def __init__(self, output_dir=None, logger=None):
        if output_dir is None:
            self.output_dir = '/tmp/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.output_dir = output_dir 
            
        self.writer = None
        self._set_writer()
    
    def _set_writer(self):
        self.writer = tf.summary.create_file_writer(self.output_dir)
        print(f"[{__class__.__name__}] set writer")
        
    def log_scalar(self, tag, value, step):
        assert self.writer is not None, ValueError(f'There writer is None')
        
        with self.writer.as_default():
            self.writer.scalar(tag, value, step)
            
    def log_histogram(self, tag, value, step):
        assert self.writer is not None, ValueError(f'There writer is None')
        
        with self.writer.as_default():
            self.writer.histogram(tag, value, step)
            
            
