import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

torch.onnx.export(model,
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        training=(
            torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL
        ),
        do_constant_folding=not train,
        input_names=["data"],
        output_names=["output"],
        dynamic_axes=(
            {
                "data": {0: "batch", 2: "height", 3: "width"},  # shape(1,3,640,640)
                "output": {0: "batch", 1: "anchors"},  # shape(1,25200,85)
            }
            if dynamic
            else None
        ),
    )