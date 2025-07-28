import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F


class LowPassFilter(nn.Module):
    def __init__(self, in_channels, filter_size, padding, padding_mode, filter_scale=1):
        super().__init__()
        self.filter_size = filter_size
        self.padding_mode = padding_mode
        self.padding = padding
        self.channels = in_channels

        if(self.filter_size == 1):
            a = np.array([[	1.,]])
        elif(self.filter_size==2):
            a = np.array([[	1., 1.]])
        elif(self.filter_size==3):
            a = np.array([[	1., 2., 1.]])
        elif(self.filter_size==4):
            a = np.array([[	1., 3., 3., 1.]])
        elif(self.filter_size==5):
            a = np.array([[	1., 4., 6., 4., 1.]])
        elif(self.filter_size==6):
            a = np.array([[	1., 5., 10., 10., 5., 1.]])
        elif(self.filter_size==7):
            a = np.array([[	1., 6., 15., 20., 15., 6., 1.]])
        else:
            raise ValueError('Filter size must be 1-7', self.filter_size)

        filt = a * a.T
        filt = torch.Tensor(filt/np.sum(filt))
        filt *= (filter_scale**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat(self.channels, 1, 1, 1))

        _allowed_padding = ('valid', 'same')
        if self.padding == 'valid':
           self.pad_tuple = None
        elif self.padding == 'same':
            _pad = (self.filter_size - 1)/2
            _pad_l = int(np.floor(_pad))
            _pad_r = int(np.ceil(_pad))
            self.pad_tuple = (_pad_l, _pad_r, _pad_l, _pad_r)
        else:
            raise ValueError(f'padding must be one of {_allowed_padding}', self.padding)

    def extra_repr(self):
        return ("in_channels={in_channels}, filter_size={filter_size}, padding={padding}, "
                "padding_mode={padding_mode}".format(in_channels=self.channels,
                filter_size=self.filter_size, padding=self.padding, padding_mode=self.padding_mode))

    def forward(self, inp):
        if self.padding != 'valid':
            inp = F.pad(inp, self.pad_tuple, self.padding_mode)
        return F.conv2d(inp, self.filt, groups=inp.shape[1])


if __name__ == '__main__':
    lpf = LowPassFilter(in_channels=64, filter_size=3, padding='same', padding_mode='circular')
    
    inputs = torch.randn((2, 32, 224, 224))
    
    outputs = lpf(inputs)
    print(outputs.shape)