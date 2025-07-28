import yaml 
import os.path as osp
from abc import *
import cv2
import numpy as np

try:
    from ultralytics import YOLO 
except Exception as error:
    print("There has been error to import YOLO from ultralytics: {error}")
    import subprocess as sp
    sp.run(['pip', 'install', 'ultralytics'])
    from ultralytics import YOLO 
    
from bytetrack.byte_tracker import BYTETracker
from tracktrack.tracker import TrackTracker

class BaseTracker:
    def __init__(self, config_file):

        assert osp.exists(config_file), ValueError(f"There is no such config-file at {config_file}")        
        with open(config_file) as yf:
            self._config = yaml.load(yf, Loader=yaml.FullLoader)

        self._set()
        
    @property
    def config(self):
        return self._config 
    
    def _set(self):
        self.set_detector()
        self.set_tracker()
        
    def set_detector(self):
        self._detector = YOLO(self._config['detector']['model_name']) 
    
    def set_tracker(self):
        if self._config['tracker']['type'] == 'BYTETracker':
            self._tracker = BYTETracker(**self._config['tracker'])
            self.track = self.track_bytetrack
        elif self._config['tracker']['type'] == 'TrackTracker':
            self._tracker = TrackTracker(**self._config['tracker'])
            self.track = self.track_tracktrack
        else:
            raise ValueError(f"There is no such tracker type: {self._config['tracker']['type']}")
        
        
    # @abstractmethod    
    # def track(self, image):
    #     pass

    # def track_bytetrack(self, image):
    #     '''
    #         return tracked_outputs: list of 3 components
    #             1. list of tlwh
    #             2. list of id
    #             3. list of conf
    #     '''
    #     if osp.isfile(image): # image or path
    #         if isinstance(image, str):
    #             if not osp.isfile(image):
    #                 raise FileNotFoundError(f"Image path not found: {image}")
    #             image = cv2.imread(image)
    #         elif isinstance(image, np.ndarray):
    #             image = image
    #         else:
    #             raise TypeError(f"Input must be a file path (str) or an image (numpy.ndarray), but got {type(image)}")
            
    #         dets = self._detector(image, verbose=False, save=False,
    #                               conf=self.config['detector']['confidence'], 
    #                               iou=self.config['detector']['iou'],
    #                               imgsz=(self.config['detector']['height'], self.config['detector']['width']),)[0]
                            
    #         boxes = dets.boxes
    #         detections = []
    #         for idx, (cls, conf, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.xyxy)):
    #             detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])

    #         if len(detections) != 0:
    #             tracked_outputs = self._tracker.track(np.array(detections), 
    #                                             (self.config['detector']['height'], self.config['detector']['width']), 
    #                                             (image.shape[0], image.shape[1]),
    #                                             aspect_ratio_thresh=self.config['post']['aspect_ratio'], 
    #                                             min_box_area=self.config['post']['min_box_area'])

    #     return tracked_outputs
    
    def track_bytetrack(self, image_path_or_array):
            
        # --- 최종 수정된 로직 ---
        # 입력값이 문자열(경로)인지, Numpy 배열(이미지)인지 확인
        if isinstance(image_path_or_array, str):
            if not osp.isfile(image_path_or_array):
                raise FileNotFoundError(f"Image path not found: {image_path_or_array}")
            image = cv2.imread(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise TypeError(f"Input must be a file path (str) or an image (numpy.ndarray), but got {type(image_path_or_array)}")
        # --- 로직 수정 끝 ---

        # 이하 코드는 동일합니다.
        dets_results = self._detector(image, verbose=False,
                                    conf=self.config['detector']['confidence'],
                                    iou=self.config['detector']['iou'],
                                    imgsz=(self.config['detector']['height'], self.config['detector']['width']))[0]

        boxes = dets_results.boxes
        detections = []
        for xyxy, conf in zip(boxes.xyxy, boxes.conf):
            detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])

        tracked_stracks = []
        if len(detections) > 0:
            tracked_stracks = self._tracker.update(np.array(detections), (image.shape[0], image.shape[1]), (image.shape[0], image.shape[1]))
        
        # tracker.py 파일과의 호환성을 위해 3개의 리스트로 변환하여 반환
        tlwhs = []
        ids = []
        scores = []
        for track in tracked_stracks:
            if not track.is_activated(): continue
            tlwhs.append(track.tlwh)
            ids.append(track.track_id)
            scores.append(track.score)
            
        return [tlwhs, ids, scores]
    
        
    def track_tracktrack(self, image):
        if osp.isfile(image):
            image = cv2.imread(image)
            dets = self._detector(image, verbose=False, save=False,
                                  conf=self.config['detector']['confidence'], 
                                  iou=self.config['detector']['iou'],
                                  imgsz=(self.config['detector']['height'], self.config['detector']['width']),
                                  )[0]
                            
            boxes = dets.boxes
            detections = []
            for idx, (cls, conf, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.xyxy)):
                detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])

            if len(detections) != 0:
                if len(detections) != 0:        
                    tracked_outputs = self._tracker.update(np.array(detections), np.array(detections))
                else:
                    tracked_outputs = self._tracker.update_without_detections()
                    
            # Filter out the results
            x1y1whs, track_ids, scores = [], [], []
            for t in tracked_outputs:
                # Check aspect ratio
                if t.x1y1wh[2] / t.x1y1wh[3] > self.config['post']['aspect_ratio']:
                    continue

                # Check track id, minimum box area
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > self.config['post']['min_box_area']:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)
                    
        return [x1y1whs, track_ids, scores]
                