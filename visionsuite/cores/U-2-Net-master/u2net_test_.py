import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import os.path as osp

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
import cv2

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = cv2.cvtColor(predict_np*255, cv2.COLOR_BGR2RGB) 
    img_name = image_name.split(os.sep)[-1]
    image = cv2.imread(image_name)
    filename = osp.split(osp.splitext(image_name)[0])[-1]
    mask = cv2.imread(osp.join(osp.split(image_name)[0], '../masks', filename + ".png"))
    imo = cv2.resize(im, (image.shape[1],image.shape[0]))


    # Image.fromarray(predict_np*255).convert('RGB')
    # img_name = image_name.split(os.sep)[-1]
    # image = io.imread(image_name)
    # imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    cv2.imwrite(os.path.join(d_dir, imidx + '.png'), np.hstack([image, mask, imo]))

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    data_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/split_mask_dataset_one_class'
    image_dir = os.path.join(data_dir, 'val/images')
    prediction_dir = '/HDD/etc/outputs/u2net/3_24_16_5_51/test/'
    model_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SOD/crop/weights/u2net_bce_itr_23244_train_0.024273_tar_0.001608.pth'

    conf_threshold = 0.5
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([
                                            # RescaleT(320),
                                            ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        weights = torch.load(model_dir)
        new_weights = {key.replace("module.", ""): value for key, value in weights.items()}
        net.load_state_dict(new_weights)
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    with torch.no_grad():
        for i_test, (data_test, fname) in enumerate(test_salobj_dataloader):

            print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            pred = torch.where(pred < conf_threshold, torch.tensor(0.), pred)
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test],pred,prediction_dir)

            del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
