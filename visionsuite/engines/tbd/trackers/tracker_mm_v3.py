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


def get_vehicle_position_mm(xyxy, homography_matrix):
    x1, y1, x2, y2 = xyxy
    bottom_center_pixel = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
    
    real_world_point = cv2.perspectiveTransform(bottom_center_pixel.reshape(-1, 1, 2), homography_matrix)
    
    return real_world_point[0][0][0]

if __name__ == '__main__':
    output_dir = 'E:\\KIA_Vehicle_Tracking_Final_YOLO_BBox'
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = 'E:\\Experiment_Dataset\\7. KIA_Vehicle_Tracking\\Tilt_전진'
    image_format = 'bmp'
    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    tracker = Tracker(config_file=config_file)
    
    # --- 호모그래피 설정 ---
    real_world_points = np.float32([[0, 0], [8000, 0], [0, 1000], [8000, 1000]])
    pixel_points = np.float32([[0, 1040], [2448, 1040], [0, 1265], [2448, 1265]])
    homography_matrix, _ = cv2.findHomography(pixel_points, real_world_points)
    print("→→→→ Homography Matrix Calculated.")
    # ----------------------
    
    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))
    start_positions_mm = {}

    for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        img_array = np.fromfile(img_file, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None: continue

        visualization_img = image.copy()

        detector_config = tracker.config['detector']
        pure_yolo_results = tracker._detector(image, verbose=False,
                                              conf=detector_config['confidence'],
                                              iou=detector_config['iou'],
                                              imgsz=(detector_config['height'], detector_config['width']))[0]
        
        tracked_outputs = tracker.track(image)

        if tracked_outputs and len(tracked_outputs) > 0 and len(tracked_outputs[0]) > 0:
            tracked_tlwhs, tracked_ids, _ = tracked_outputs
            
            for yolo_box in pure_yolo_results.boxes:
                yolo_xyxy = yolo_box.xyxy[0].cpu().numpy()
                yolo_tlwh = [yolo_xyxy[0], yolo_xyxy[1], yolo_xyxy[2] - yolo_xyxy[0], yolo_xyxy[3] - yolo_xyxy[1]]
                
                best_iou = 0.5
                matched_id = -1
                for i, track_tlwh in enumerate(tracked_tlwhs):
                    iou = iou_distance([yolo_tlwh], [track_tlwh])[0][0]
                    if iou > best_iou:
                        best_iou = iou
                        matched_id = tracked_ids[i]

                if matched_id != -1:
                    current_pos_mm = get_vehicle_position_mm(yolo_xyxy, homography_matrix)

                    if matched_id not in start_positions_mm:
                        start_positions_mm[matched_id] = current_pos_mm
                    
                    moved_distance_mm = abs(current_pos_mm - start_positions_mm[matched_id])

                    x1, y1, x2, y2 = map(int, yolo_xyxy)
                    cv2.rectangle(visualization_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    info_text = f"ID:{matched_id} Moved:{moved_distance_mm / 1000:.2f}m"
                    cv2.putText(visualization_img, info_text, 
                                (x1, y1 - 10), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2,
                                color=(0, 255, 0), lineType=cv2.LINE_AA)

        output_path = osp.join(output_dir, filename + '.jpg')
        is_success, im_buf_arr = cv2.imencode(".jpg", visualization_img)
        if is_success:
            im_buf_arr.tofile(output_path)
        
        def iou_distance(atlbrs, btlbrs):
            ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
            if ious.size == 0:
                return ious

            for i, a in enumerate(atlbrs):
                for j, b in enumerate(btlbrs):
                    ax1, ay1, aw, ah = a
                    ax2, ay2 = ax1 + aw, ay1 + ah
                    bx1, by1, bw, bh = b
                    bx2, by2 = bx1 + bw, by1 + bh
                    
                    inter_x1 = max(ax1, bx1)
                    inter_y1 = max(ay1, by1)
                    inter_x2 = min(ax2, bx2)
                    inter_y2 = min(ay2, by2)
                    
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    
                    inter_area = inter_w * inter_h
                    a_area = aw * ah
                    b_area = bw * bh
                    union_area = a_area + b_area - inter_area
                    
                    ious[i, j] = inter_area / union_area if union_area > 0 else 0
            return ious