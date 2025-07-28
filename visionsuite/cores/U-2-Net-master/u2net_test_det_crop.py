import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import json
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import os.path as osp
from copy import deepcopy

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
import cv2
from visionsuite.utils.dataset.formats.labelme.utils import add_labelme_element, init_labelme_json, get_points_from_image
import imgviz 

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def main():

    # --------- 1. get image path and name ---------
    mode = '1st'
    model_name='u2net'
    model_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SOD/crop/weights/u2net_bce_itr_23244_train_0.024273_tar_0.001608.pth'
    detection = 'define'
    # detection = 'yolov12_xl'
    for defect in ['오염', '딥러닝', '경계성', 'repeated_ng', 'repeated_ok']:
        
        for order in [1, 2, 3]:
            if mode == '1st':
                img_dir = f'/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data/OUTER_shot0{order}'
            else:
                img_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/data/{order}'
            if order == 1:
                labelme_dir = f'/HDD/etc/repeatablility/talos2/{mode}/benchmark/{detection}/{defect}/exp/labels'
                output_dir = f'/HDD/etc/repeatablility/talos2/{mode}/benchmark/{detection}_sod/{defect}/exp/labels'
            else:
                labelme_dir = f'/HDD/etc/repeatablility/talos2/{mode}/benchmark/{detection}/{defect}/exp{order}/labels'
                output_dir = f'/HDD/etc/repeatablility/talos2/{mode}/benchmark/{detection}_sod/{defect}/exp{order}/labels'
                
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)        

            roi = [220, 60, 1340, 828]
            offset = 320
            classes = ['background', "CHAMFER_MARK", "LINE", "MARK"]
            class2idx = {val: key for key, val in enumerate(classes)}
            color_map = imgviz.label_colormap()[1:len(class2idx) + 1 + 1]
            conf_threshold = 0.6
            contour_thres = 10
            vis = True

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

            json_files = glob.glob(osp.join(labelme_dir, '*.json'))
            with torch.no_grad():
                for json_file in tqdm(json_files, desc=f'Order {defect}-{order}: '):
                    
                    filename = osp.split(osp.splitext(json_file)[0])[-1]
                    if mode == '1st':
                        img_file = osp.join(img_dir, f'{filename}_Outer', '1_image.bmp')
                    else:
                        img_file = osp.join(img_dir, filename, '1_image.bmp')
                    
                    assert osp.exists(img_file), ValueError(f'There is no image: {img_file}')
                    
                    
                    img = cv2.imread(img_file)
                    img_h, img_w, _ = img.shape
                    if vis:
                        vis_pred = np.zeros((img_h, img_w, len(class2idx)), dtype=np.uint8)
                    roi_img = img[roi[1]:roi[3], roi[0]:roi[2]]
                    with open(json_file, 'r') as jf:
                        anns = json.load(jf)
                    
                    _labelme = init_labelme_json(f'{filename}.bmp', img_w, img_h)
                    
                    for ann_id, ann in enumerate(anns['shapes']):
                            
                        label = ann['label']
                        if ann['shape_type'] in ['watershed']:
                            continue
                        
                        xs, ys = [], []
                        for point in ann['points']:
                            if point[0] >= roi[0] and point[0] <= roi[2]:
                                xs.append(point[0] - roi[0])
                            elif point[0] < roi[0]:
                                xs.append(0)
                            elif point[0] > roi[2]:
                                xs.append(roi[2] - roi[0])
                            else:
                                raise RuntimeError(
                                    f"Not considered x: point is {point} and roi is {roi}"
                                )

                            if point[1] >= roi[1] and point[1] <= roi[3]:
                                ys.append(point[1] - roi[1])
                            elif point[1] < roi[1]:
                                ys.append(0)
                            elif point[1] > roi[3]:
                                ys.append(roi[3] - roi[1])
                            else:
                                raise RuntimeError(
                                    f"Not considered y: point is {point} and roi is {roi}"
                                )
                                
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        obj_width = max(xs) - min(xs)
                        obj_height = max(ys) - min(ys)
                        case = None
                        if isinstance(offset, float) and offset <= 1 and offset >= 0:
                            offset_x, offset_y = (x2 - x1) * offset, (y2 - y1) * offset
                            case = 1
                            do_resize = False
                        elif isinstance(offset, (tuple, list)):
                            assert len(offset) == 2, ValueError(f"Offset for crop must be 2 values as x and y")
                            offset_x, offset_y = offset[0], offset[1]
                            case = 1
                            do_resize = True
                        elif isinstance(offset, int):
                            offset_x, offset_y = offset, offset
                            case = 2
                            do_resize = True
                            
                            if offset_x < obj_width:
                                offset_x = obj_width + 20
                                
                            if offset_y < obj_height:
                                offset_y = obj_height + 20
                            
                        else:
                            raise NotImplementedError(f"NOT yet Considered: {offset}")
                            
                            
                        if case == 1:
                            x1 = x1 - offset_x if x1 - offset_x >= 0 else 0
                            y1 = y1 - offset_y if y1 - offset_y >= 0 else 0
                            x2 = x2 + offset_x if x1 + offset_x <= roi[2] - roi[0] else roi[2] - roi[0]
                            y2 = y2 + offset_y if y1 + offset_y <= roi[3] - roi[1] else roi[3] - roi[1]
                        elif case == 2:
                            cx, cy = (x2 - x1)/2 + x1, (y2 - y1)/2 + y1
                            x1 = cx - offset_x/2 
                            y1 = cy - offset_y/2 
                            x2 = cx + offset_x/2 
                            y2 = cy + offset_y/2 
                            
                            if x1 < 0:
                                x2 -= x1
                                x1 = 0
                            
                            if x2 > ( roi[2] - roi[0]):
                                x1 -= (x2 - ( roi[2] - roi[0]))
                                x2 = ( roi[2] - roi[0]) 
                            
                            if y1 < 0:
                                y2 -= y1
                                y1 = 0
                            
                            if y2 > (roi[3] - roi[1]):
                                y1 -= (y2 - (roi[3] - roi[1]))
                                y2 = (roi[3] - roi[1]) 
                            
                        crop_img = roi_img[int(y1):int(y2), int(x1):int(x2)]
                                            
                        if do_resize:
                            ori_crop_h, ori_crop_w, _ = crop_img.shape
                            if ori_crop_w != offset or ori_crop_h != offset:
                                print("eklwajflkwjefklawjefklawjefklajweklf: ", ori_crop_h, ori_crop_h)
                            ori_crop = deepcopy(crop_img)
                            
                            crop_img = cv2.resize(crop_img, (int(offset), int(offset)))

                        assert crop_img.shape[:2] == (int(offset), int(offset))
                                    
                                        
                        tmpImg = np.zeros((crop_img.shape[0],crop_img.shape[1],3))
                        crop_img = crop_img/np.max(crop_img)
                        if crop_img.shape[2]==1:
                            tmpImg[:,:,0] = (crop_img[:,:,0]-0.485)/0.229
                            tmpImg[:,:,1] = (crop_img[:,:,0]-0.485)/0.229
                            tmpImg[:,:,2] = (crop_img[:,:,0]-0.485)/0.229
                        else:
                            tmpImg[:,:,0] = (crop_img[:,:,0]-0.485)/0.229
                            tmpImg[:,:,1] = (crop_img[:,:,1]-0.456)/0.224
                            tmpImg[:,:,2] = (crop_img[:,:,2]-0.406)/0.225
                                        
                        torch_img = torch.from_numpy(tmpImg.transpose((2, 0, 1))).float().unsqueeze(0)

                        if torch.cuda.is_available():
                            torch_img = Variable(torch_img.cuda())
                        else:
                            torch_img = Variable(torch_img)

                        d1,d2,d3,d4,d5,d6,d7= net(torch_img)

                        # normalization
                        pred = d1[:,0,:,:]
                        pred = torch.where(pred < conf_threshold, torch.tensor(0.), torch.tensor(1.))
                        pred = normPRED(pred)

                        predict = pred
                        predict = predict.squeeze()
                        predict_np = predict.cpu().data.numpy()
                        if do_resize:
                            predict_np = cv2.resize(predict_np, (ori_crop_w, ori_crop_h))
                        
                        contours, _ = cv2.findContours(predict_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        points = []
                        for contour in contours:
                            if len(contour) < contour_thres:
                                pass
                            else:
                                epsilon = 0.001 * cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].tolist()
                                points.append(approx)
                                
                        for point in points:
                            new_points = []    
                            for _point in point:
                                new_points.append([int(_point[0] + x1 + roi[0]), int(_point[1] + y1 + roi[1])]) 
                        
                            if len(new_points) < 3:
                                continue
                        
                            if vis:
                                mask = vis_pred[:, :, class2idx[label]].copy()  # view → copy로 바꿔서 C-contiguous 보장
                                cv2.fillPoly(mask, [np.array(new_points )], color=(1))
                                vis_pred[:, :, class2idx[label]] = mask

                            _labelme = add_labelme_element(_labelme, shape_type='polygon', 
                                                label=label, 
                                                points=new_points)
                        
                        # cv2.imwrite(os.path.join(output_dir, filename + '.png'), np.hstack([crop_img, im]))
                        del d1,d2,d3,d4,d5,d6,d7
                    
                    with open(os.path.join(output_dir, filename + ".json"), "w") as jsf:
                        json.dump(_labelme, jsf)    
                
                    if vis:
                        vis_dir = osp.join(output_dir, '../vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_pred = vis_pred.argmax(-1)
                        vis_pred = cv2.addWeighted(img, 0.6, color_map[vis_pred], 0.4, 0)
                        cv2.imwrite(osp.join(vis_dir, filename + '.png'), np.hstack([img, vis_pred]))

if __name__ == "__main__":
    main()
