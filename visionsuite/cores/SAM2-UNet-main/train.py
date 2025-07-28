import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", default="/HDD/weights/sam2/sam2_hiera_large.pt")
parser.add_argument("--train_image_path", default='/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/datasets/split_mask_patch_dataset/train/images/')
parser.add_argument("--train_mask_path", default='/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/datasets/split_mask_patch_dataset/train/masks/')
parser.add_argument('--output_dir', default="/HDD/datasets/projects/LX/24.11.28_2/datasets_wo_vertical/outputs/sam2_unet/train")
parser.add_argument("--epoch", type=int, default=200, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(args):  
    import os.path as osp  
    import imgviz
    import cv2
    classes = ['background', 'timber', 'screw']
    color_map = imgviz.label_colormap()[1:len(classes) + 1]
    freq_vis_val = 20
    freq_save_model = 20
    vis_dir = osp.join(args.output_dir, 'vis')
    if not osp.exists(vis_dir):
        os.mkdir(vis_dir),
        
    weights_dir = osp.join(args.output_dir, 'weights')
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)

    dataset = FullDataset(args.train_image_path, args.train_mask_path, (512, 512), mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
            # if epoch != 0 and epoch%freq_vis_val == 0:
            if True:
                _vis_dir = osp.join(vis_dir, str(epoch))
                if not osp.exists(_vis_dir):
                    os.mkdir(_vis_dir)
                for pred in pred0:
                    predicted = pred0.argmax(dim=0)
    
                    for idx, pred in enumerate(predicted):
                        _pred = pred.cpu().detach().numpy()
                        _pred = color_map[_pred]
                        cv2.imwrite(osp.join(_vis_dir, batch['filename'][idx] + '.png'), _pred)
                        # cv2.imwrite(osp.join(_vis_dir, f'{idx}.png'), _pred)
                    
                
        scheduler.step()
        if epoch != 0 and (epoch%freq_save_model == 0 or (epoch+1) == args.epoch):
            torch.save(model.state_dict(), os.path.join(weights_dir, 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(weights_dir, 'SAM2-UNet-%d.pth'% (epoch + 1)))


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)