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

# weights = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/dinov2/weights/80.pth'
weights = None
epochs = 301
batch_size = 8
model_height = 980
model_width = 980
input_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/datasets/split_mask_patch_dataset'
output_dir = '/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/dinov2'
freq_save_model = 20
freq_vis_val = 20
vis_dir = osp.join(output_dir, 'vis')
if not osp.exists(vis_dir):
    os.mkdir(vis_dir)
    
weights_dir = osp.join(output_dir, 'weights')
if not osp.exists(weights_dir):
    os.mkdir(weights_dir)

import albumentations as A

ADE_MEAN = tuple(np.array([123.675, 116.280, 103.530]) / 255)
ADE_STD = tuple(np.array([58.395, 57.120, 57.375]) / 255)

train_transform = A.Compose([
    # hadded an issue with an image being too small to crop, PadIfNeeded didn't help...
    # if anyone knows why this is happening I'm happy to read why
    # A.PadIfNeeded(min_height=512, min_width=512),
    # A.RandomResizedCrop(height=512, width=512),
    A.Resize(width=model_width, height=model_height),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

val_transform = A.Compose([
    A.Resize(width=model_width, height=model_height),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])


train_dataset = SegmentationDataset(Dataset(input_dir, 'train'), transform=train_transform)
val_dataset = SegmentationDataset(Dataset(input_dir, 'val'), transform=val_transform)
     
     
pixel_values, target, original_image, original_segmentation_map, _ = train_dataset[3]
print(pixel_values.shape)
print(target.shape)

import numpy as np
import matplotlib.pyplot as plt

id2label = {0: 'background', 1: 'timber', 2: 'screw'}
id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}
color_map = imgviz.label_colormap()[1:len(id2label) + 1]

print([id2label[id] for id in np.unique(original_segmentation_map).tolist()])


def visualize_map(image, segmentation_map):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in id2color.items():
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()

from torch.utils.data import DataLoader

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]
    batch["filename"] = [i[4] for i in inputs]

    return batch

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v,torch.Tensor):
    print(k,v.shape)
    

from PIL import Image

unnormalized_image = (batch["pixel_values"][0].numpy() * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
unnormalized_image = Image.fromarray(unnormalized_image)
unnormalized_image
     
# visualize_map(unnormalized_image, batch["labels"][0].numpy())

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
      loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
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
    print(f">>> LOADED model: {weights}")
    

for name, param in model.named_parameters():
  if name.startswith("dinov2"):
    param.requires_grad = False
    
outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
print(outputs.logits.shape)
print(outputs.loss)


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
model.train()

for epoch in range(epochs):
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # forward pass
        outputs = model(pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        # evaluate
        with torch.no_grad():
            predicted = outputs.logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            if epoch != 0 and epoch%freq_vis_val == 0:
                _vis_dir = osp.join(vis_dir, str(epoch))
                if not osp.exists(_vis_dir):
                    os.mkdir(_vis_dir)
                    
                for idx, pred in enumerate(predicted):
                    _pred = pred.cpu().detach().numpy()
                    _pred = color_map[_pred]
                    cv2.imwrite(osp.join(_vis_dir, batch['filename'][idx] + '.png'), _pred)
                    
                

        # let's print loss and metrics every 100 batches
        if idx % 100 == 0:
            metrics = metric.compute(num_labels=len(id2label),
                                    ignore_index=0,
                                    reduce_labels=False,
            )

            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])
        
        if epoch != 0 and epoch%freq_save_model == 0:
            torch.save(model.state_dict(), osp.join(weights_dir, f"{epoch}.pth"))
        