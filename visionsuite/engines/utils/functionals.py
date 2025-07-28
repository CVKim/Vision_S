import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def denormalize(x, mean=MEAN, std=STD):
    x *= np.array(std)
    x += np.array(mean)
    x *= 255
    x = x.astype(np.uint8)
    
    return x