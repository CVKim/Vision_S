import imgviz
import numpy as np
import cv2
import os.path as osp
import os 
import torch

def vis_by_batch(output, target, image, fname, epoch, batch_idx, output_dir, denormalize=None,
              input_channel=3, origin = (25,25), font = cv2.FONT_HERSHEY_SIMPLEX):
    
    for step_idx, (pred, mask, image, fname) in enumerate(zip(output, target, image, fname)):
        vis_by_step(pred, mask, image, fname, epoch, batch_idx, step_idx, output_dir, denormalize=denormalize, 
                    input_channel=input_channel, origin=origin)
        
                
def vis_by_step(pred, mask, image, fname, epoch, batch_idx, step_idx, output_dir, denormalize=None, 
                input_channel=3, origin = (25,25), font = cv2.FONT_HERSHEY_SIMPLEX):
    pred = torch.nn.functional.softmax(pred, dim=0)
    pred = torch.argmax(pred, dim=0)
    pred = pred.detach().float().to('cpu')
    pred = pred.numpy()
    
    image = image.to('cpu')
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    image = denormalize(image)
    color_map = imgviz.label_colormap(256)
    mask = mask.cpu().detach()
    mask = color_map[mask.numpy().astype(np.uint8)].astype(np.uint8)
    pred = color_map[pred.astype(np.uint8)].astype(np.uint8)
    
    pred = cv2.addWeighted(image, 0.1, pred, 0.9, 0)
    mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)

    text1 = np.zeros((50, image.shape[1], input_channel), np.uint8)
    text2 = np.zeros((50, image.shape[1], input_channel), np.uint8)
    text3 = np.zeros((50, image.shape[1], input_channel), np.uint8)
    cv2.putText(text1, "(a) original", origin, font, 0.6, (255,255,255), 1)
    cv2.putText(text2, "(b) ground truth" , origin, font, 0.6, (255,255,255), 1)
    cv2.putText(text3, "(c) predicted" , origin, font, 0.6, (255,255,255), 1)

    image = cv2.vconcat([text1, image])
    mask = cv2.vconcat([text2, mask])
    pred = cv2.vconcat([text3, pred.astype(np.uint8)])

    res = cv2.hconcat([image, mask, pred])
    vis_dir = osp.join(output_dir, str(epoch))
    if not osp.exists(vis_dir):
        os.mkdir(vis_dir)
    cv2.imwrite(osp.join(vis_dir, f"{fname}_{batch_idx}_{step_idx}.png"), res)