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
    
    output_dir = 'E:\\KIA_Vehicle_Tracking_Diagnosis'
    os.makedirs(output_dir, exist_ok=True)
    input_dir = 'E:\\Experiment_Dataset\\7. KIA_Vehicle_Tracking\\Tilt_전진'
    image_format = 'bmp'
    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    tracker = Tracker(config_file=config_file)
    
    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))

    for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        img_array = np.fromfile(img_file, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None: continue

        visualization_img = image.copy()

        # --- 1. 순수 YOLO 탐지 결과 확인 (파란색 박스) ---
        detector_config = tracker.config['detector']
        pure_yolo_results = tracker._detector(image, verbose=False,
                                              conf=detector_config['confidence'],
                                              iou=detector_config['iou'],
                                              imgsz=(detector_config['height'], detector_config['width']))[0]
        
        for box in pure_yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(visualization_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # 파란색
            cv2.putText(visualization_img, "YOLO", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- 2. ByteTrack 최종 추적 결과 확인 (초록색 박스) ---
        tracked_outputs = tracker.track(image)
        
        if tracked_outputs and len(tracked_outputs) > 0 and len(tracked_outputs[0]) > 0:
            for tlwh, id, score in zip(*tracked_outputs):
                x, y, w, h = map(int, tlwh)
                cv2.rectangle(visualization_img, (x, y), (x + w, y + h), (0, 255, 0), 2) # 초록색
                cv2.putText(visualization_img, f"Track:{id}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        output_path = osp.join(output_dir, filename + '.jpg')
        # 원본 image가 아닌, 그림이 그려진 visualization_img를 저장합니다.
        is_success, im_buf_arr = cv2.imencode(".jpg", visualization_img)
        if is_success:
            im_buf_arr.tofile(output_path)
        # -------------------------