from .cifar_dataset import CifarDataset
from .load_data import load_data, image_folder #TODO: to be deprecated
from .directory_dataset import DirectoryDataset

__all__ = ['load_data', 'image_folder', 'DirectoryDataset', 'CifarDataset']