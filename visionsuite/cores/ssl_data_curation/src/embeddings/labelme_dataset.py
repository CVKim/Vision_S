import os.path as osp 
import glob
from PIL import Image 

class LabelmeDataset:
    def __init__(self, root, transform, image_format='bmp'):
        self.transform = transform
        # self.roi = [220, 60, 1340, 828]
        self.img_files = glob.glob(osp.join(root, f'*.{image_format}'))
        print(f"There are {len(self.img_files)} image files")
        
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = Image.open(img_file)
        # img = img.crop((self.roi[0], self.roi[1], self.roi[2], self.roi[3]))

        if self.transform:
            img = self.transform(img)
            
        return img, osp.split(osp.splitext(img_file)[0])[-1]
        
        
    def __len__(self):
        return len(self.img_files)