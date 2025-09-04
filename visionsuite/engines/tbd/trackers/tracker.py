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
    
    output_dir = 'E:\\KIA_Vehicle_Tracking_back_tilt'
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = 'E:\\Experiment_Dataset\\7. KIA_Vehicle_Tracking\\Tilt_전진'
    image_format = 'bmp'
    
    frame_skip = 4

    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    tracker = Tracker(config_file=config_file)
    
    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))

    indices_to_process = set(range(0, len(img_files), frame_skip))
    
    if len(img_files) > 0:
        indices_to_process.add(len(img_files) - 1)
    # ---------------------------------------
    
    for i in tqdm(sorted(list(indices_to_process))):
        img_file = img_files[i]
        
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        
        img_array = np.fromfile(img_file, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"\n 경고: 이미지를 로드할 수 없습니다: {img_file}")
            continue

        tracked_outputs = tracker.track(img_file)
        
        if tracked_outputs and len(tracked_outputs) > 0 and len(tracked_outputs[0]) > 0:
            for tlwh, id, score in zip(*tracked_outputs):
                cv2.rectangle(image, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                                (0, 0, 255), 2)
                cv2.putText(image, f"{id}_{score:0.1f}", 
                            (int(tlwh[0]), int(tlwh[1]) - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2,
                            color=(0, 0, 255), lineType=3
                            )
        output_path = osp.join(output_dir, filename + '.jpg')
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        if is_success:
            im_buf_arr.tofile(output_path)