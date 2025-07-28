from abc import abstractmethod
from visionsuite.engines.segmentation.utils.registry import DATASETS
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule


@DATASETS.register()
class BaseDataset(BaseOOPModule):
    def __init__(self, transform=None):
        super().__init__()
        self.args = None
        
        self._transform = transform
        self.train_dataset = None
        self.val_dataset = None
        
        self.train_sampler = None 
        self.val_sampler = None
        
        self.label2index = None
        self.index2label = None
        self.num_classes = None
        self.classes = None
        
    @property 
    def transform(self):
        return self._transform 
    
    @transform.setter
    def transform(self, val):
        self._transform = val
        
    def build(self, load=True, *args, **kwargs):
        super().build(*args, **kwargs)
        
        if 'classes' in kwargs:
            self.classes = kwargs['classes']
            
        if load:
            self._load()
    
    
    @BaseOOPModule.track_status
    def _load(self):
        self.load_dataset()
        self.load_sampler()

    @BaseOOPModule.track_status
    @abstractmethod
    def load_dataset(self):
        pass

    @BaseOOPModule.track_status
    def load_sampler(self):
        import torch 
        from visionsuite.engines.classification.utils.registry import SAMPLERS

        if self.args['distributed']['use']:
            if self.args['sampler']['type']:
                self.train_sampler = SAMPLERS.get(self.args['sampler']['type'])(self.train_dataset, shuffle=True, repetitions=self.args['sampler']['reps'])
            else:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
            
            
