import os 
import os.path as osp 
from glob import glob 
from ultralytics import YOLO 
import cv2


def track_video(model, source, det_conf, det_iou, tracker, output_dir):
    results = model.track(source=source,
                      conf=det_conf, iou=det_iou,
                      tracker=tracker, save=True, save_txt=False,
                      project=output_dir,
                      persist=True, verbose=False,
                    )
    
    return results

def track_images(model, source, det_conf, det_iou, tracker, output_dir, num_frames, image_format='jpg'):
    img_files = sorted(glob(osp.join(source, f'*.{image_format}')))

    for img_file in img_files[:num_frames]:
        filename = osp.split(osp.splitext(img_file)[0])[-1]    
        res = model.track(img_file, conf=det_conf, iou=det_iou, tracker=tracker, persist=persist, save=False, verbose=False)
        
        names = res[0].names
        boxes = res[0].boxes
        
        img = cv2.imread(img_file)
        for idx, (cls, conf, id, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.id, boxes.xyxy)):
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                        (0, 0, 255), 2)
            # cv2.putText(img, f"{id}_{names[int(cls)]}_{conf:0.2f}", 
            cv2.putText(img, f"{id}_{names[int(cls)]}", 
                        (int(xyxy[0]), int(xyxy[1] - 10)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(0, 0, 255), lineType=3
                    )
        
        cv2.imwrite(osp.join(output_dir, filename + '.jpg'), img)
        
if __name__ == '__main__':
    
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent

    output_dir = '/HDD/etc/outputs/tracking/ultralytics'
    os.makedirs(output_dir, exist_ok=True)

    model_name = 'yolo11n.pt'
    source = str(ROOT / 'data/video.mp4')
    # source = str(ROOT / 'data/mot17')
    image_format = 'jpg'
    tracker = str(ROOT / 'cfg/trackers/bytetrack.yaml')
    # tracker = '/HDD/_projects/github/VisionSuite/visionsuite/cores/ultralytics_track/cfg/trackers/botsort.yaml'
    det_conf = 0.3
    det_iou = 0.5
    persist = True

    model = YOLO(model_name)

    if 'mp4' in source:
        outputs = track_video(model, source, det_conf, det_iou, tracker, output_dir)
    elif osp.isdir(source):
        output_dir = osp.join(output_dir, 'tracked_images')
        os.makedirs(output_dir, exist_ok=True)
        num_frames = 120
        track_images(model, source, det_conf, det_iou, tracker, output_dir, num_frames, image_format=image_format)

