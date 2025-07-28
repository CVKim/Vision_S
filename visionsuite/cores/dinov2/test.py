import numpy as np
from torch.utils.data import Dataset
import torch

import os.path as osp
import glob
import cv2
import os
import imgviz

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

weights = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/dinov2/train/weights/80.pth'
batch_size = 1
model_height = 980
model_width = 980
input_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/datasets/split_mask_patch_dataset'
output_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/dinov2/test'
import albumentations as A

ADE_MEAN = tuple(np.array([123.675, 116.280, 103.530]) / 255)
ADE_STD = tuple(np.array([58.395, 57.120, 57.375]) / 255)
val_transform = A.Compose([
    A.Resize(width=model_width, height=model_height),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])


val_dataset = SegmentationDataset(Dataset(input_dir, 'val'), transform=val_transform)
import numpy as np

id2label = {0: 'background', 1: 'timber', 2: 'screw'}
id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}
color_map = imgviz.label_colormap()[1:len(id2label) + 1]

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

import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.dinov2 = Dinov2Model(config)
    self.classifier = LinearClassifier(config.hidden_size, 70, 70, config.num_labels)

  def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
    outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    # get the patch embeddings - so we exclude the CLS token
    patch_embeddings = outputs.last_hidden_state[:,1:,:]

    # convert to logits and upsample to the size of the pixel values
    logits = self.classifier(patch_embeddings)
    logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

    loss = None
    if labels is not None:
      # important: we're going to use 0 here as ignore index instead of the default -100
      # as we don't want the model to learn to predict background
      loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
      loss = loss_fct(logits.squeeze(), labels.squeeze())

    return SemanticSegmenterOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))

if weights is not None and osp.exists(weights): 
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    

import evaluate

metric = evaluate.load("mean_iou")

from torch.optim import AdamW
from tqdm.auto import tqdm

# training hyperparameters
# NOTE: I've just put some random ones here, not optimized at all
# feel free to experiment, see also DINOv2 paper
learning_rate = 5e-5

optimizer = AdamW(model.parameters(), lr=learning_rate)

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
        predicted = outputs.logits.argmax(dim=1)

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
    