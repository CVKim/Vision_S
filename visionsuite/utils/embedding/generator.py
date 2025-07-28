import os.path as osp 
import os 
import numpy as np 
import torch 

from .loaders import ImageIterator

class EmbeddingGenerator:
    def __init__(self, input_dir, output_dir, 
                        batch_size=4, device='cuda', seed=42,
                        phis='dinov2'):
        
        self._input_dir = input_dir
        self._output_dir = output_dir 
        self._batch_size = batch_size
        self._device = device 
        self._seed = seed 
        self._phis = phis 
        
        
    def _set_model(self):
        
        if self._phis == 'dinov2':
            torch.hub.set_dir(osp.join(self._output_dir, "checkpoints/dinov2"))
            # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            self._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(self._device)
            self._model.eval()
            print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self._model.parameters()]):,}")
            preprocess = None
        
        
        
    def _set_dataloader(self):
        self._dataloader = ImageIterator(input_dir=self._input_dir, transform=None, batch_size=self.)
        