import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

img_size = 224
x = torch.randn(8, 3, img_size, img_size) # bchw
patch_size = 16
in_channels = 3
batch_size = 8

# ----------------------
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print(patches.shape)

# ----------------------
emb_size = patch_size*patch_size*in_channels

projection = nn.Sequential(
    nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), # behw
    Rearrange('b e h w -> b (h w) e') # b(hw)e
)

print(projection(x).shape)

# ----------------------

projected_x = projection(x)
print("projected x: ", projected_x.shape)

cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
print(f'cls token: {cls_token.shape}')
print(f'positions: {positions.shape}')

cls_tokens = repeat(cls_token, '() n e -> b n e', b=batch_size)
print(f'cls tokens: {cls_tokens.shape}')

cat_x = torch.cat([cls_tokens, projected_x], dim=1)
cat_x += positions 
print(f'cat_x: ', cat_x.shape)

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------