import os
import torch
import os.path as osp 
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

from PIL import Image 
from skimage import io
import cv2

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def normPREDv2(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(data_dir, pred, d_dir, names, masks):

    for predict, name, mask in zip(pred, names, masks):    
        predict_np = predict.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy()

        im = cv2.cvtColor(predict_np*255, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask_np*255, cv2.COLOR_BGR2RGB)
        image = cv2.imread(osp.join(data_dir, f'val/images/{name}.png'))
        # image = cv2.imread(osp.join(data_dir, f'DUTS-TE/DUTS-TE-Image/{name}.jpg'))
        imo = cv2.resize(im, (image.shape[1],image.shape[0]))
        mask = cv2.resize(mask, (image.shape[1],image.shape[0]))

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        cv2.imwrite(osp.join(d_dir, name + '.png'), np.hstack([image, mask, imo]))

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	return loss0, loss1, loss2, loss3, loss4, loss5, loss6


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

# data_dir = '/HDD/datasets/public/DUTS/'
# tra_image_dir = os.path.join('DUTS-TR/DUTS-TR-Image')
# tra_label_dir = os.path.join('DUTS-TR/DUTS-TR-Mask')
# val_image_dir = os.path.join('DUTS-TE/DUTS-TE-Image')
# val_label_dir = os.path.join('DUTS-TE/DUTS-TE-Mask')
# image_ext = '.jpg'
# data_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_mask_dataset_one_class'
data_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_mask_dataset_one_class_crop'
tra_image_dir = os.path.join('train/images')
tra_label_dir = os.path.join('train/masks')
val_image_dir = os.path.join('val/images')
val_label_dir = os.path.join('val/masks')
image_ext = '.png'

label_ext = '.png'

output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SOD/'
now = datetime.now()
output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}')
if not osp.exists(output_dir):
    os.makedirs(output_dir)
    
weights_dir = osp.join(output_dir, 'weights')
if not osp.exists(weights_dir):
    os.makedirs(weights_dir)

val_dir = osp.join(output_dir, 'val')
if not osp.exists(val_dir):
    os.makedirs(val_dir)

debug_dir = osp.join(output_dir, 'debug')
if not osp.exists(debug_dir):
    os.makedirs(debug_dir)

epoch_num = 300
batch_size_train = 96
batch_size_val = 1
train_num = 0
val_num = 0

#### ================================ train 
tra_img_name_list = glob.glob(os.path.join(data_dir,  tra_image_dir, f'*{image_ext}'))

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(os.path.join(data_dir,  tra_label_dir, f'{imidx}{label_ext}'))

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

### ================================ val. dataset

val_img_name_list = glob.glob(os.path.join(data_dir,  val_image_dir, f'*{image_ext}'))

val_lbl_name_list = []
for img_path in val_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	val_lbl_name_list.append(os.path.join(data_dir,  val_label_dir, f'{imidx}{label_ext}'))

print("---")
print("val images: ", len(val_img_name_list))
print("val labels: ", len(val_lbl_name_list))
print("---")

val_num = len(val_img_name_list)


salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        # RescaleT(320),
        # RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)


val_salobj_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        # RescaleT(320),
        # RandomCrop(288),
        ToTensorLab(flag=0)]))
val_salobj_dataloader = DataLoader(val_salobj_dataset, batch_size=batch_size_val, shuffle=True, num_workers=1)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)
    
weights = '/HDD/weights/u2net/u2net.pth'
try:
    net.load_state_dict(torch.load(weights))    
except Exception as error:
    raise Exception(f"There has been error when loading weights: {weights} b/c {error}")
print(f"SUCCESSFULLY LOADED MODEL: {weights}")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
  
if torch.cuda.is_available():
    net.to('cuda:0')
    
# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_freq = 9 # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, (data, names) in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        if epoch == 1:
            _debug_dir = osp.join(debug_dir, 'train')
            if not osp.exists(_debug_dir):
                os.mkdir(_debug_dir)
    
            for _input, label, name in zip(inputs, labels, names):
                if random.random() < 0.1:
                    _input = _input.permute((1, 2, 0)).cpu().detach().numpy()
                    label = label.permute((1, 2, 0)).cpu().detach().numpy()
                    label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    
                    cv2.imwrite(osp.join(_debug_dir, name + '.png'), np.hstack([normPREDv2(_input)*255, normPREDv2(label)*255]))

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss0, loss1, loss2, loss3, loss4, loss5, loss6 = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        loss2 = loss0
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        
        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss

        print(f"[epoch: {epoch + 1:3d}/{epoch_num:3d}, batch: {((i + 1) * batch_size_train):5d}/{train_num:5d}, ite: {ite_num}] train loss: {running_loss / ite_num4val:.6f}, tar: {running_tar_loss / ite_num4val:.6f}, l0: {loss0.data.item():3f}, l1: {loss1.data.item():3f}, l2: {loss2.data.item():3f}, l3: {loss3.data.item():3f}, l4: {loss4.data.item():3f}, l5: {loss5.data.item():3f}, l6: {loss6.data.item():3f}", 
              end="\r")
    
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss


    if epoch == 1:
        for j, (val_data, val_names) in enumerate(val_salobj_dataloader):
            inputs, labels = val_data['image'], val_data['label']

            _debug_dir = osp.join(debug_dir, 'val')
            if not osp.exists(_debug_dir):
                os.mkdir(_debug_dir)

            for _input, label, name in zip(inputs, labels, val_names):
                if random.random() < 0.1:
                    _input = _input.permute((1, 2, 0)).cpu().detach().numpy()
                    label = label.permute((1, 2, 0)).cpu().detach().numpy()
                    label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    
                    cv2.imwrite(osp.join(_debug_dir, name + '.png'), np.hstack([normPREDv2(_input)*255, normPREDv2(label)*255]))


    if epoch % save_freq == 0:
        net.eval()
        print()
        with torch.no_grad():
            for j, (val_data, val_names) in enumerate(val_salobj_dataloader):
                inputs, labels = val_data['image'], val_data['label']
                
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss0, loss1, loss2, loss3, loss4, loss5, loss6 = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                loss2 = loss0
                loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                        
                print(f"[VAL] [epoch: {epoch + 1:3d}/{epoch_num:3d}, batch: {((j + 1) * batch_size_train):5d}/{train_num:5d}, ite: {ite_num}] train loss: {running_loss / ite_num4val:.6f}, tar: {running_tar_loss / ite_num4val:.6f}, l0: {loss0.data.item():3f}, l1: {loss1.data.item():3f}, l2: {loss2.data.item():3f}, l3: {loss3.data.item():3f}, l4: {loss4.data.item():3f}, l5: {loss5.data.item():3f}, l6: {loss6.data.item():3f}", end="\r")
            
                _val_dir = osp.join(val_dir, str(epoch))
                if not osp.exists(_val_dir):
                    os.mkdir(_val_dir)
                
                pred = d1[:,0,:,:]
                pred = normPRED(pred)
                                
                save_output(data_dir, pred, _val_dir, val_names, labels[:, 0, :, :])
        
        
        torch.save(net.module.state_dict(), os.path.join(weights_dir, model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)))
    running_loss = 0.0
    running_tar_loss = 0.0
    net.train()  # resume train
    ite_num4val = 0
        
    print("\n===============================================================================================")
