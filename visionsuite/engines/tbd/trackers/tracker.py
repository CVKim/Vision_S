import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob 
    
from pathlib import Path
from base_tracker import BaseTracker

class Tracker(BaseTracker):
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

if __name__ == '__main__':
    
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent
    
    output_dir = 'E:\\MOT'
    # output_dir = '/HDD/etc/outputs/tracking/tracktrack/yolo'
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = 'E:\\Experiment_Dataset\\MOT17\\train\\MOT17-13-DPM\\img1'
    image_format = 'jpg'
    num_frames = 2
    
    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    # config_file = '/HDD/_projects/github/VisionSuite/visionsuite/engines/tbd/trackers/configs/tracktrack.yaml'
    tracker = Tracker(config_file=config_file)
    
    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))
    for img_file in tqdm(img_files[:num_frames]):
    # for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]  
        image = cv2.imread(img_file)
        tracked_outputs = tracker.track(img_file)
        
        for tlwh, id, score in zip(*tracked_outputs):
            cv2.rectangle(image, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                        (0, 0, 255), 2)
            cv2.putText(image, f"{id}_{score:0.1f}", 
                        (int(tlwh[0]), int(tlwh[1] - 10)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2,
                        color=(0, 0, 255), lineType=3
                    )
            
        cv2.imwrite(osp.join(output_dir, filename + '.jpg'), image)