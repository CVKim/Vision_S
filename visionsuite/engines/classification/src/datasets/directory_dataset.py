import os.path as osp

from visionsuite.engines.classification.utils.registry import DATASETS
from visionsuite.engines.classification.src.datasets.base_dataset import BaseDataset


@DATASETS.register()
class DirectoryDataset(BaseDataset):
    def __init__(self, name="DirectoryDataset", transform=None):
        if transform is None:
            mean=(0.485, 0.456, 0.406)
            std=(0.229, 0.224, 0.225)
                
            import torchvision.transforms as transforms
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
            
        super().__init__(name=name, transform=transform)

    def load_dataset(self, train_folder_name='train', val_folder_name='val'):
        super().load_dataset()
        
        self.train_dataset =  DATASETS.get(self.args['load_dataset']['type'], case_sensitive=self.args['load_dataset']['case_sensitive'])(osp.join(self.args['input_dir'], train_folder_name), self._transform)
        self.log_info(f"LOADED train_dataset: {self.args['load_dataset']['type']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- input_dir: {self.args['input_dir']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- number of images: {len(self.train_dataset)}", self.build.__name__, __class__.__name__)
        self.log_info(f"- transforms: TODO", self.build.__name__, __class__.__name__)
        
        self.val_dataset =  DATASETS.get(self.args['load_dataset']['type'], case_sensitive=self.args['load_dataset']['case_sensitive'])(osp.join(self.args['input_dir'], val_folder_name), self._transform)
        self.log_info(f"LOADED val_dataset: {self.args['load_dataset']['type']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- input_dir: {self.args['input_dir']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- number of images:: {len(self.val_dataset)}", self.build.__name__, __class__.__name__)
        self.log_info(f"- transforms: TODO", self.build.__name__, __class__.__name__)
        
        self.label2index = {index: label for index, label in enumerate(self.train_dataset.classes)}
        self.index2label = {label: index for index, label in enumerate(self.train_dataset.classes)}
        self.classes = self.train_dataset.classes
        self.num_classes = len(self.train_dataset.classes)
        self.log_info(f"- label2index: {self.label2index}", self.load_dataset.__name__, __class__.__name__)

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

