import os.path as osp 
import glob
from PIL import Image 

class LabelmeDataset:
    def __init__(self, root, transform, image_format='bmp'):
        self.transform = transform
        
        self.img_files = glob.glob(osp.join(root, f'*.{image_format}'))
        print(f"There are {len(self.img_files)} image files")
        
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = Image.open(img_file)
        target = None
        
        if self.transform:
            img = self.transform(img)
            
        return img, target
        