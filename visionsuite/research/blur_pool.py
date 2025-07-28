import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        self.stride = stride

        # Create a fixed blur kernel (e.g., Gaussian-like)
        if filt_size == 3:
            kernel = torch.tensor([1., 2., 1.])
        elif filt_size == 5:
            kernel = torch.tensor([1., 4., 6., 4., 1.])
        else:
            raise NotImplementedError()

        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.groups = channels

    def forward(self, x):
        # Padding to keep size consistent
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x, self.kernel, stride=self.stride, padding=0, groups=self.groups)


if __name__ == '__main__':
    import numpy as np 
    import scipy 
    
    def checkerboard(size=64, num_checks=8):
        row_check = np.kron([[1, 0] * (num_checks // 2), [0, 1] * (num_checks // 2)] * (num_checks // 2), 
                            np.ones((size // num_checks, size // num_checks)))
        return row_check



    image = checkerboard()
    print(f'image shape: {image.shape}')
    blurpool = BlurPool(3)
    
    torch_image = torch.tensor(image)
    torch_image = torch_image.repeat(3, 1, 1)[None, ...]
    
    output = blurpool(torch_image)
    