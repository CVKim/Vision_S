import os
import os.path as osp
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

def make_parser():
    # Initialization
    parser = argparse.ArgumentParser("Track")

    # Data args
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--data_path", type=str, default="/HDD/datasets/public/MOT17/test/")
    parser.add_argument("--pickle_path", type=str, default="/HDD/etc/outputs/tracking/tracktrack/1. det/mot17_test_0.95_original.pickle")
    parser.add_argument("--output_dir", type=str, default="/HDD/etc/outputs/tracking/tracktrack/2. det_feat")
    parser.add_argument("--output_filename", type=str, default="mot17_test_0.95_original.pickle")
    parser.add_argument("--config_path", type=str, default=str(ROOT / "configs/MOT17/sbs_S50.yml"))
    parser.add_argument("--weight_path", type=str, default="/HDD/weights/tracktrack/fastreid/mot17_sbs_S50.pth")

    # Else
    parser.add_argument("--seed", type=float, default=10000)

    return parser


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Get encoder
    embedder = EmbeddingComputer(config_path=args.config_path, weight_path=args.weight_path)

    # Read detection
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Feature extraction
    for vid_name in detections.keys():
        for frame_id in detections[vid_name].keys():
            # If there is no detection
            if detections[vid_name][frame_id] is None:
                continue

            # Read image
            if 'MOT' in args.data_path:
                img = cv2.imread(args.data_path + vid_name + '/img1/%06d.jpg' % frame_id)
            else:
                img = cv2.imread(args.data_path + vid_name + '/img1/%08d.jpg' % frame_id)

            # Get detection
            detection = detections[vid_name][frame_id]

            # Get features
            if detection is not None:
                embedding = embedder.compute_embedding(img, detection[:, :4])
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)

            # Logging
            print(vid_name, frame_id, flush=True)

    # Save
    with open(osp.join(args.output_dir, args.output_filename), 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
