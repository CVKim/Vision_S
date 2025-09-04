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
    """B-Box의 tlwh를 받아, 호모그래피 변환 후 현실 세계의 X좌표(mm)를 반환합니다."""
    x, y, w, h = tlwh
    bottom_center_pixel = np.array([[(x + w / 2), (y + h)]], dtype=np.float32)
    real_world_point = cv2.perspectiveTransform(bottom_center_pixel.reshape(-1, 1, 2), homography_matrix)
    return real_world_point[0][0][0]

if __name__ == '__main__':
    
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent
    
    output_dir = 'E:\\KIA_Vehicle_Tracking_back_tilt_mm_ver_03'
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = 'E:\\Experiment_Dataset\\7. KIA_Vehicle_Tracking\\Tilt_전진'
    image_format = 'bmp'
    
    frame_skip = 4

    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    tracker = Tracker(config_file=config_file)
    
    # --- 🔴 호모그래피 설정 (프로그램 시작 시 1회 계산) ---
    # 1. 레일의 실제 mm 좌표 (dst_points): 가로 8m, 세로 1m
    real_world_points = np.float32([
        [0, 0],       # 좌상단 (현실 기준점)
        [8000, 0],    # 우상단
        [0, 1000],    # 좌하단
        [8000, 1000]  # 우하단
    ])

    # 2. 레일의 픽셀 좌표 (src_points) - 제공해주신 값
    pixel_points = np.float32([
        [0, 1040],    # 좌상단
        [2448, 1040], # 우상단
        [0, 1265],    # 좌하단
        [2448, 1265]  # 우하단
    ])

    # 3. 호모그래피 행렬 계산
    homography_matrix, _ = cv2.findHomography(pixel_points, real_world_points)
    print("✅ Homography Matrix Calculated.")
    # ----------------------------------------------------
    
    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))
    
    indices_to_process = set(range(0, len(img_files), frame_skip))
    if len(img_files) > 0:
        indices_to_process.add(len(img_files) - 1)
    
    start_positions_mm = {}

    for i in tqdm(sorted(list(indices_to_process))):
        img_file = img_files[i]
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        
        # 한글 경로가 포함된 이미지를 올바르게 읽기
        img_array = np.fromfile(img_file, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"\n 경고: 이미지를 로드할 수 없습니다: {img_file}")
            continue

        # 🔴 실제 ByteTrack 모델을 통해 추적 결과 얻기 (시뮬레이션 아님)
        tracked_outputs = tracker.track(img_file)
        
        # 추적 결과가 있을 경우에만 시각화
        if tracked_outputs and len(tracked_outputs) > 0 and len(tracked_outputs[0]) > 0:
            for tlwh, id, score in zip(*tracked_outputs):
                
                # 1. 현재 차량의 실제 mm 위치를 계산
                current_pos_mm = get_vehicle_position_mm(tlwh, homography_matrix)

                # 2. 해당 ID의 시작 위치가 없다면 현재 위치를 시작 위치로 저장
                if id not in start_positions_mm:
                    start_positions_mm[id] = current_pos_mm
                
                # 3. 실제 이동 거리(mm) 계산
                moved_distance_mm = abs(current_pos_mm - start_positions_mm[id])

                # B-Box 그리기
                cv2.rectangle(image, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                                (0, 255, 0), 2)
                
                # 정보 텍스트 수정: 계산된 이동 거리를 미터 단위로 표시
                info_text = f"ID:{id} Moved:{moved_distance_mm / 1000:.2f}m"
                cv2.putText(image, info_text, 
                            (int(tlwh[0]), int(tlwh[1]) - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2,
                            color=(0, 255, 0), lineType=cv2.LINE_AA)
                
        # 한글 경로가 포함된 파일명으로 결과 저장
        output_path = osp.join(output_dir, filename + '.jpg')
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        if is_success:
            im_buf_arr.tofile(output_path)