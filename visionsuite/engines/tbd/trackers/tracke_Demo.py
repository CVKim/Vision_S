# FILE: tracker.py (수정된 최종본)

import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path

# base_tracker.py는 이전과 동일하게 사용합니다.
from base_tracker import BaseTracker

class Tracker(BaseTracker):
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent

    # --- 1. 경로 및 설정 초기화 ---
    output_dir = osp.join(ROOT, 'output_with_distance_fix')
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = 'E:\\Experiment_Dataset\\MOT17\\train\\MOT17-13-DPM\\img1'
    config_file = 'E:\\DL_SW\\VisionSuite\\visionsuite\\engines\\tbd\\trackers\\configs\\bytetrack.yaml'
    image_format = 'jpg'

    tracker = Tracker(config_file=config_file)

    # --- 2. 거리 측정을 위한 초기화 ---
    K = np.array([[369.538, 0, 317.742], [0, 369.251, 237.099], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.399484, 0.19908, 0.000213683, 0.000477909, -0.0588417])
    
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_frame_gray = None
    prev_frame_data = {}  # {id: {'kp': keypoints, 'desc': descriptors}}
    tracked_car_info = {} # {id: {'distance': float}}

    SCALE_FACTOR = 20.0
    TRIGGER_DISTANCE = 10.0
    INITIAL_DISTANCE = 50.0

    # --- 3. 메인 처리 루프 ---
    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))
    for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        image = cv2.imread(img_file)
        if image is None: continue

        # 렌즈 왜곡 보정 (새로 추가)
        undistorted_image = cv2.undistort(image, K, dist_coeffs, None, K)
        current_frame_gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        
        # 1. 트래킹 수행
        # tracked_outputs 형식: [ [tlwh1,...], [id1,...], [score1,...] ]
        tracked_outputs = tracker.track(image)
        if not tracked_outputs or not tracked_outputs[0]:
            # 추적된 객체가 없으면 다음 프레임으로
            prev_frame_gray = image
            prev_frame_data = {} # 이전 데이터 초기화
            cv2.imwrite(osp.join(output_dir, filename + '.jpg'), image)
            continue
            
        tracked_tlwhs, tracked_ids, tracked_scores = tracked_outputs
        
        current_frame_data = {}

        # 2. 현재 프레임의 추적된 객체들에 대해 특징점 계산
        for tlwh, id, score in zip(tracked_tlwhs, tracked_ids, tracked_scores):
            x, y, w, h = map(int, tlwh)
            roi = image[y:y+h, x:x+w]
            if roi.size == 0: continue

            kp, desc = orb.detectAndCompute(roi, None)

            if kp:
                for point in kp: point.pt = (point.pt[0] + x, point.pt[1] + y)
                current_frame_data[id] = {'kp': kp, 'desc': desc}
            
            if id not in tracked_car_info:
                tracked_car_info[id] = {'distance': INITIAL_DISTANCE}

        # 3. 이전 프레임과 비교하여 거리 계산
        if prev_frame_gray is not None and prev_frame_data:
            for id, data in current_frame_data.items():
                if id in prev_frame_data:
                    prev_data = prev_frame_data[id]
                    matches = bf.match(prev_data['desc'], data['desc'])
                    good_matches = sorted(matches, key=lambda x: x.distance)[:30]

                    if len(good_matches) > 8:
                        pts1 = np.float32([prev_data['kp'][m.queryIdx].pt for m in good_matches])
                        pts2 = np.float32([data['kp'][m.trainIdx].pt for m in good_matches])
                        
                        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        if E is None: continue

                        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
                        
                        distance_moved = np.linalg.norm(t) * SCALE_FACTOR
                        tracked_car_info[id]['distance'] -= distance_moved
                        if tracked_car_info[id]['distance'] < 0: tracked_car_info[id]['distance'] = 0

        # 4. 결과 시각화
        for tlwh, id, score in zip(tracked_tlwhs, tracked_ids, tracked_scores):
            color = (0, 255, 0)
            distance = tracked_car_info.get(id, {}).get('distance', INITIAL_DISTANCE)
            
            if distance <= TRIGGER_DISTANCE:
                color = (0, 0, 255)

            cv2.rectangle(image, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), color, 2)
            info_text = f"ID:{id} Dist:{distance:.1f}m"
            cv2.putText(image, info_text, (int(tlwh[0]), int(tlwh[1] - 10)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2, color=color, lineType=cv2.LINE_AA)

        cv2.imwrite(osp.join(output_dir, filename + '.jpg'), image)

        prev_frame_gray = current_frame_gray
        prev_frame_data = current_frame_data