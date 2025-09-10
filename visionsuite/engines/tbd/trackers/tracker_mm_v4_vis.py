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

def get_vehicle_position_mm(tlwh, homography_matrix):
    x, y, w, h = tlwh
    bottom_center_pixel = np.array([[(x + w / 2), (y + h)]], dtype=np.float32)
    real_world_point = cv2.perspectiveTransform(bottom_center_pixel.reshape(-1, 1, 2), homography_matrix)
    return real_world_point[0][0]

if __name__ == '__main__':
    
    output_dir = 'E:\\KIA_Vehicle_Tracking_Visualization_02'
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = 'E:\\Experiment_Dataset\\7. KIA_Vehicle_Tracking\\Tilt_전진'
    image_format = 'bmp'
    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    tracker = Tracker(config_file=config_file)
    
    # --- 호모그래피 설정 ---
    real_world_points = np.float32([[0, 0], [8000, 0], [0, 1000], [8000, 1000]])
    pixel_points = np.float32([[0, 1060], [2448, 1020], [0, 1280], [2448, 1235]])
    homography_matrix, _ = cv2.findHomography(pixel_points, real_world_points)
    print("✅ Homography Matrix Calculated.")
    # ----------------------

    bev_width_px = 800
    bev_height_px = int(bev_width_px * (1000 / 8000))
    bev_corners = np.float32([[0, 0], [bev_width_px, 0], [0, bev_height_px], [bev_width_px, bev_height_px]])
    bev_matrix = cv2.getPerspectiveTransform(pixel_points, bev_corners)
    # ---------------------------------------------

    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))
    start_positions_mm = {}
    
    last_known_positions_mm = {} # {id: (last_x_mm, last_y_mm), ...}
    # ---------------------------------------

    for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        img_array = np.fromfile(img_file, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None: continue

        visualization_img = image.copy()
        bev_image = cv2.warpPerspective(image, bev_matrix, (bev_width_px, bev_height_px))
        tracked_outputs = tracker.track(image)
        
        if tracked_outputs and len(tracked_outputs) > 0 and len(tracked_outputs[0]) > 0:
            for tlwh, id, score in zip(*tracked_outputs):
                current_pos_mm_x, current_pos_mm_y = get_vehicle_position_mm(tlwh, homography_matrix)

                if id not in start_positions_mm:
                    start_positions_mm[id] = current_pos_mm_x
                
                moved_distance_mm = abs(current_pos_mm_x - start_positions_mm[id])

                last_known_positions_mm[id] = (current_pos_mm_x, current_pos_mm_y)

                cv2.rectangle(visualization_img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                                (0, 255, 0), 2)
                
                info_text = f"ID:{id} Moved:{moved_distance_mm / 1000:.2f}m"
                cv2.putText(visualization_img, info_text, (int(tlwh[0]), int(tlwh[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if last_known_positions_mm:
            for id, pos_mm in last_known_positions_mm.items():
                pos_x_mm, pos_y_mm = pos_mm
                bev_point_x = int(pos_x_mm / 8000 * bev_width_px)
                bev_point_y = int(pos_y_mm / 1000 * bev_height_px)
                
                cv2.circle(bev_image, (bev_point_x, bev_point_y), 5, (0, 0, 255), -1)
                cv2.putText(bev_image, str(id), (bev_point_x + 10, bev_point_y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # -----------------------------------------------------------------

        bev_image_resized = cv2.resize(bev_image, (visualization_img.shape[1], int(visualization_img.shape[1] * (bev_height_px/bev_width_px))))
        final_image = cv2.vconcat([visualization_img, bev_image_resized])
        
        output_path = osp.join(output_dir, filename + '.jpg')
        is_success, im_buf_arr = cv2.imencode(".jpg", final_image)
        if is_success:
            im_buf_arr.tofile(output_path)