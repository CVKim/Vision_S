import cv2 
import numpy as np
import os.path as osp 
import torch 
import imgviz

        
def save_validation(model, device, dataset, num_classes, epoch, output_dir, denormalize=False, input_channel=3, \
                        image_channel_order='bgr', validation_image_idxes_list=[], color_map=imgviz.label_colormap(50)):
    model.eval()
    origin = 25,25
    font = cv2.FONT_HERSHEY_SIMPLEX

    # if len(validation_image_idxes_list) == 0:
    #     validation_image_idxes_list = range(0, len(dataset))

    total_idx = 1
    for batch in dataset:
        if len(batch) == 3:
            image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
        else: 
            image, mask = batch[0].detach(), batch[1].detach()
            fname = None           
        image = image.to(device, dtype=torch.float32)
        # image = image.to(device)
        image = image.unsqueeze(0)
        preds = model(image)
        if isinstance(preds, dict):
            preds = preds['out']
        elif isinstance(preds, list):
            preds = preds[0]
        
        preds = preds[0]
        preds = torch.nn.functional.softmax(preds, dim=0)
        preds = torch.argmax(preds, dim=0)
        preds = preds.detach().float().to('cpu')
        # preds.apply_(lambda x: t2l[x])
        preds = preds.numpy()

        image = image.to('cpu')[0]
        image = image.numpy()
        image = image.transpose((1, 2, 0))
        if denormalize:
            image = denormalize(image)
        # image = image.astype(np.uint8)
        mask = color_map[mask.numpy().astype(np.uint8)].astype(np.uint8)
        if input_channel == 3:
            if image_channel_order == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image_channel_order == 'bgr':
                pass
            else:
                raise ValueError(f"There is no such image_channel_order({image_channel_order})")

            preds = color_map[preds.astype(np.uint8)].astype(np.uint8)
        elif input_channel == 1:
            preds = color_map[preds.astype(np.uint8)].astype(np.uint8)

        else:
            raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

        preds = cv2.addWeighted(image, 0.1, preds, 0.9, 0)
        mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)

        text1 = np.zeros((50, image.shape[1], input_channel), np.uint8)
        text2 = np.zeros((50, image.shape[1], input_channel), np.uint8)
        text3 = np.zeros((50, image.shape[1], input_channel), np.uint8)
        cv2.putText(text1, "(a) original", origin, font, 0.6, (255,255,255), 1)
        cv2.putText(text2, "(b) ground truth" , origin, font, 0.6, (255,255,255), 1)
        cv2.putText(text3, "(c) predicted" , origin, font, 0.6, (255,255,255), 1)

        image = cv2.vconcat([text1, image])
        mask = cv2.vconcat([text2, mask])
        preds = cv2.vconcat([text3, preds.astype(np.uint8)])

        res = cv2.hconcat([image, mask, preds])
        cv2.imwrite(osp.join(output_dir, str(epoch) + "_" + fname + '_{}.png'.format(total_idx)), res)
        total_idx += 1