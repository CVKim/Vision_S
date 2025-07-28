import numpy as np
from torch.utils.data import Dataset
import torch

import os.path as osp
import glob
import cv2
import os
import imgviz
from model import LinearProbingSam2

class Dataset:
    def __init__(self, input_dir, mode, img_ext='png', mask_ext='png'):
        self.input_dir = input_dir 
        self.mode = mode
        self.img_files = glob.glob(osp.join(input_dir, mode, f"images/*.{img_ext}"))
        
    def __len__(self):
        return len(self.img_files)    
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        mask_file = osp.join(self.input_dir, self.mode, 'masks', filename + '.png')
        
        assert osp.exists(img_file), ValueError(f'There is no such image file: {img_file}')
        assert osp.exists(mask_file), ValueError(f'There is no such mask file: {mask_file}')
        
        
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file, 0)
        
        
        return {"image": img, "label": mask, 'filename': filename}
        
        

class SegmentationDataset(Dataset):
  def __init__(self, dataset, transform):
    self.dataset = dataset
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    original_image = np.array(item["image"])
    original_segmentation_map = np.array(item["label"])

    transformed = self.transform(image=original_image, mask=original_segmentation_map)
    image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

    # convert to C, H, W
    image = image.permute(2,0,1)

    return image, target, original_image, original_segmentation_map, item['filename']

weights = "/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/sam2/tiny/train/weights/300.pth"
batch_size = 1
model_height = 1024
model_width = 1024
input_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/datasets/split_mask_patch_dataset'
output_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/sam2/tiny/test'
if not osp.exists(output_dir):
    os.mkdir(output_dir)

import albumentations as A

ADE_MEAN = tuple(np.array([123.675, 116.280, 103.530]) / 255)
ADE_STD = tuple(np.array([58.395, 57.120, 57.375]) / 255)

val_transform = A.Compose([
    A.Resize(width=model_width, height=model_height),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])

val_dataset = SegmentationDataset(Dataset(input_dir, 'val'), transform=val_transform)
     
     
import numpy as np
import matplotlib.pyplot as plt

id2label = {0: 'background', 1: 'timber', 2: 'screw'}
id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}
color_map = imgviz.label_colormap(50)[1:][1:len(id2label) + 1]

from torch.utils.data import DataLoader

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]
    batch["filename"] = [i[4] for i in inputs]

    return batch

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

sam2_checkpoint = "/HDD/weights/sam2/sam2_hiera_tiny.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
# model_cfg = "/HDD/sam2-main/sam2/configs/sam2/sam2_hiera_s.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]
model_cfg = "sam2_hiera_t.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

# sam2_checkpoint = "/HDD/weights/sam2/sam2_hiera_large.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
# # model_cfg = "/HDD/sam2-main/sam2/configs/sam2/sam2_hiera_s.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]
# model_cfg = "sam2_hiera_l.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]


model = LinearProbingSam2(256, num_classes=len(id2label), sam2_checkpoint=sam2_checkpoint, model_cfg=model_cfg)

if weights is not None and osp.exists(weights): 
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    print(f">>> LOADED model: {weights}")
    

import evaluate

metric = evaluate.load("mean_iou")

from tqdm.auto import tqdm
# put model on GPU (set runtime to GPU in Google Colab)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# put model in training mode
model.eval()

with torch.no_grad():
    for idx, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # forward pass
        outputs = model(pixel_values)

        predicted = outputs['logits'].argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        for idx, pred in enumerate(predicted):
            _pred = pred.cpu().detach().numpy()
            _pred = color_map[_pred]
            cv2.imwrite(osp.join(output_dir, batch['filename'][idx] + '.png'), _pred)
                
                

    metrics = metric.compute(num_labels=len(id2label),
                            ignore_index=0,
                            reduce_labels=False,
    )

    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])
        