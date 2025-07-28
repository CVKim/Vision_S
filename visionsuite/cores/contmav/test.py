import argparse
from datetime import datetime
import json
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim
from src.args_test import ArgumentParser
from src.build_model import build_model
from src import utils
from src.prepare_data import prepare_data
import imgviz 
import cv2

from torchmetrics import JaccardIndex as IoU


def parse_args():
    parser = ArgumentParser(
        description="Open-World Semantic Segmentation (Training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_common_args()
    args = parser.parse_args()
    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        print(
            f"Notice: adapting learning rate to {args.lr} because provided "
            f"batch size differs from default batch size of 8."
        )

    return args


def train_main():
    args = parse_args()

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(
        args.results_dir, args.dataset, f"{args.id}", f"{training_starttime}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "confusion_matrices"), exist_ok=True)

    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, "argsv.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(args, ckpt_dir)

    train_loader, valid_loader, _ = data_loaders

    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != "None":
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting
        )
    else:
        class_weighting = np.ones(n_classes_without_void)
    # model building -----------------------------------------------------------
    model, device = build_model(args, n_classes=n_classes_without_void)

    # loss functions
    loss_function_train = utils.CrossEntropyLoss2d(
        weight=class_weighting, device=device
    )
    loss_objectosphere = utils.ObjectosphereLoss()
    loss_mav = utils.OWLoss(n_classes=n_classes_without_void)
    loss_contrastive = utils.ContrastiveLoss(n_classes=n_classes_without_void)

    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(
        weight_mode="linear"
    )
    pixel_sum_valid_data_weighted = np.sum(pixel_sum_valid_data * class_weighting)
    loss_function_valid = utils.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device,
    )

    train_loss = [loss_function_train, loss_objectosphere, loss_mav, loss_contrastive]
    val_loss = [loss_function_valid, loss_objectosphere, loss_mav, loss_contrastive]
    if not args.obj:
        train_loss[1] = None
        val_loss[1] = None
    if not args.mav:
        train_loss[2] = None
        val_loss[2] = None
    if not args.closs:
        train_loss[3] = None
        val_loss[3] = None
        
    if args.last_ckpt != "":
        ckpt = torch.load(args.last_ckpt)
        model.load_state_dict(ckpt['state_dict'])
        print("LOADED ckpt: ", args.last_ckpt)

    if args.load_weights != "":
        model.load_state_dict(torch.load(args.load_weights), strict=True)
        print("LOADED weights: ", args.load_weights)
        
    writer = SummaryWriter("runs/" + ckpt_dir.split(args.dataset)[-1])
    test(
        model=model,
        valid_loader=valid_loader,
        device=device,
        val_loss=val_loss,
        epoch=0,
        debug_mode=args.debug,
        writer=writer,
        classes=args.num_classes,
    )
    writer.flush()


def test(
    model,
    valid_loader,
    device,
    val_loss,
    epoch,
    writer,
    loss_function_valid_unweighted=None,
    debug_mode=False,
    classes=19,
):
    # set model to eval mode
    model.eval()

    # we want to store miou and ious for each camera
    miou = dict()
    ious = dict()

    loss_function_valid, loss_obj, loss_mav, loss_contrastive = val_loss

    # reset loss (of last validation) to zero
    loss_function_valid.reset_loss()

    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()

    compute_iou = IoU(
        task="multiclass", num_classes=classes, average="none", ignore_index=255
    ).to(device)

    mavs = None
    if loss_contrastive is not None:
        mavs = loss_mav.read()

    total_loss_obj = []
    total_loss_mav = []
    total_loss_con = []
    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.
    color_map=imgviz.label_colormap(50)

    for i, sample in enumerate(tqdm(valid_loader, desc="Valid step")):
        # copy the data to gpu
        image = sample["image"].to(device)
        target = sample["label"].long().cuda() - 1
        target[target == -1] = 255

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction_ss, prediction_ow = model(image)

            for (_image, _target, pred_ss) in zip(image, target, prediction_ss):
                vis_image = np.zeros((512, 1024*3, 3))
                
                vis_pred_ss = pred_ss.detach().cpu().numpy()
                vis_pred_ss = np.transpose(vis_pred_ss, (1, 2, 0))#.astype(np.uint8)
                vis_pred_ss = np.argmax(vis_pred_ss, axis=-1)
                vis_pred_ss = color_map[vis_pred_ss].astype(np.uint8)
                
                vis_image[:, :1024, :] = np.transpose(_image.detach().cpu().numpy(), (1, 2, 0))
                vis_image[:, 1024:1024*2, :] = color_map[_target.detach().cpu().numpy()].astype(np.uint8)
                vis_image[:, 1024*2:, :] = vis_pred_ss
                cv2.imwrite("/HDD/etc/contmav.png", vis_image)
                

            if not device.type == "cpu":
                torch.cuda.synchronize()

            compute_iou.update(prediction_ss, target.cuda())

            # compute valid loss
            loss_function_valid.add_loss_of_batch(
                prediction_ss, sample["label"].to(device)
            )

            loss_objectosphere = torch.tensor(0)
            loss_ows = torch.tensor(0)
            loss_con = torch.tensor(0)
            if loss_obj is not None:
                target_obj = sample["label"]
                target_obj[target_obj == 16] = 255
                target_obj[target_obj == 17] = 255
                target_obj[target_obj == 18] = 255
                loss_objectosphere = loss_obj(prediction_ow, sample["label"])
            total_loss_obj.append(loss_objectosphere.cpu().detach().numpy())
            if loss_mav is not None:
                loss_ows = loss_mav(prediction_ss, target.cuda(), is_train=False)
            total_loss_mav.append(loss_ows.cpu().detach().numpy())
            if loss_contrastive is not None:
                loss_con = loss_contrastive(mavs, prediction_ow, target, epoch)
            total_loss_con.append(loss_con.cpu().detach().numpy())

            if debug_mode:
                # only one batch while debugging
                break

    ious = compute_iou.compute().detach().cpu()
    miou = ious.mean()

    total_loss = (
        loss_function_valid.compute_whole_loss()
        + np.mean(total_loss_obj)
        + np.mean(total_loss_mav)
        + np.mean(total_loss_con)
    )
    print("LOSS: ", total_loss)
    print("Metrics/miou: ", miou)
    writer.add_scalar("Loss/val", total_loss, epoch)
    writer.add_scalar("Metrics/miou", miou, epoch)
    for i, iou in enumerate(ious):
        writer.add_scalar(
            "Class_metrics/iou_{}".format(i),
            torch.mean(iou),
            epoch,
        )

    return miou

    return 1 - best[0], best[1]


if __name__ == "__main__":
    train_main()
