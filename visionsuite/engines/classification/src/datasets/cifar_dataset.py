import torchvision
from visionsuite.engines.classification.utils.registry import DATASETS
from visionsuite.engines.classification.src.datasets.base_dataset import BaseDataset


@DATASETS.register()
class CifarDataset(BaseDataset):
    def __init__(self, num_classes=10, transform=None):
        if transform is None:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
                
            import torchvision.transforms as transforms
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        super().__init__(transform=transform)
        self.num_classes = num_classes
        
    def load_dataset(self):
        super().load_dataset()
        
        self.train_dataset = getattr(torchvision.datasets, f"CIFAR{self.num_classes}")(root='/tmp/data', 
                                                        train=True, download=True, transform=self._transform)
            
        self.val_dataset = getattr(torchvision.datasets, f"CIFAR{self.num_classes}")(root='/tmp/data', 
                                                        train=False, download=True, transform=self._transform)

        self.label2index = {index: label for index, label in enumerate(self.train_dataset.classes)}
        self.index2label = {label: index for index, label in enumerate(self.train_dataset.classes)}
        self.classes = self.train_dataset.classes
        print(f"label2index: {self.label2index}")


