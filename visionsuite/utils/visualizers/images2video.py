import cv2
import os
import os.path as osp
import glob

input_dir = "/HDD/etc/outputs/tracking/bytetrack/yolo"
output_dir = "/HDD/etc/outputs/tracking/bytetrack/yolo/.."
fps = 30
img_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))  

frame = cv2.imread(img_paths[0])
height, width, layers = frame.shape
size = (width, height)

out = cv2.VideoWriter(osp.join(output_dir, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for img_path in img_paths:
    img = cv2.imread(img_path)
    out.write(img)

out.release()