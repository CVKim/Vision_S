from abc import abstractmethod
from visionsuite.engines.classification.utils.registry import DATASETS
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule


@DATASETS.register()
class BaseDataset(BaseOOPModule):
    def __init__(self, name=None, transform=None, mode='train'):
        super().__init__(name=name)
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
    
    def build(self, load=True, *args, **kwargs):
        super().build(*args, **kwargs)
        
        if 'classes' in kwargs:
            self.classes = kwargs['classes']
            
        self.log_info(f"Built dataset: {self.args}", self.build.__name__, __class__.__name__)
        
        if load:
            self._load()
    
    @BaseOOPModule.track_status
    def _load(self):
        self.load_dataset()
        self.load_sampler()

    @BaseOOPModule.track_status
    @abstractmethod
    def load_dataset(self):
        self.log_info(f"Loading dataset", self.load_dataset.__name__, __class__.__name__)
        pass

    @BaseOOPModule.track_status
    def load_sampler(self):
        self.log_info(f"Loading sampler", self.load_sampler.__name__, __class__.__name__)
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
            

# def load_data(traindir, valdir, transform, args):
#     import os
#     import torch
#     import torchvision
#     import time
#     import torchvision.transforms
#     from visionsuite.engines.utils.helpers import mkdir, get_cache_path
#     from visionsuite.engines.utils.torch_utils.utils import save_on_master
#     from visionsuite.engines.classification.utils.registry import SAMPLERS

#     # Data loading code
#     print("Loading data")
    

#     st = time.time()
#     cache_path = get_cache_path(traindir)
#     if args.cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print(f"Loading dataset_train from {cache_path}")
#         # TODO: this could probably be weights_only=True
#         dataset, _ = torch.load(cache_path, weights_only=False)
#     else:
#         dataset = torchvision.datasets.ImageFolder(
#             traindir,
#             transform,
#         )
#         if args.cache_dataset:
#             print(f"Saving dataset_train to {cache_path}")
#             mkdir(os.path.dirname(cache_path))
#             save_on_master((dataset, traindir), cache_path)
#     print("Took", time.time() - st)

#     print("Loading validation data")
#     cache_path = get_cache_path(valdir)
#     if args.cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print(f"Loading dataset_test from {cache_path}")
#         # TODO: this could probably be weights_only=True
#         dataset_test, _ = torch.load(cache_path, weights_only=False)
#     else:
#         if args.model['weights']:
#             weights = torchvision.models.get_weight(args.model['weights'])
#             preprocessing = weights.transforms(antialias=True)
#             if args.backend == "tensor":
#                 preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

#         else:
#             preprocessing = transform

#         dataset_test = torchvision.datasets.ImageFolder(
#             valdir,
#             preprocessing,
#         )
#         if args.cache_dataset:
#             print(f"Saving dataset_test to {cache_path}")
#             mkdir(os.path.dirname(cache_path))
#             save_on_master((dataset_test, valdir), cache_path)

#     print("Creating data loaders")


#     train_sampler, test_sampler = SAMPLERS.get('get_samplers')(args, dataset, dataset_test)

#     return dataset, dataset_test, train_sampler, test_sampler

