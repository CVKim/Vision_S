import cv2
import os
import os.path as osp
from natsort import natsorted

input_dir = "/HDD/datasets/public/MOT17/train/MOT17-02-DPM/img1/"
output_dir = '/HDD/etc/outputs/tracking'
fps = 30
num_frames = 600

img_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
img_files = natsorted(img_files)

first_frame = cv2.imread(os.path.join(input_dir, img_files[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(osp.join(output_dir, 'video.mp4'), fourcc, 30, (width, height))

for file_name in img_files[:num_frames]:
    img_path = os.path.join(input_dir, file_name)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()