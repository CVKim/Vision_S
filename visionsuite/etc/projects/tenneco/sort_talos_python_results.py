from glob import glob 
import os 
import os.path as osp 
from tqdm import tqdm
from shutil import copyfile

def find_jpg_file(root_dir, target_filename):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == target_filename and file.endswith('.jpg'):
                return os.path.join(root, file)
    return None



talos_dir = '/DeepLearning/research/data/benchmarks/benchmarks_production/tenneco/repeatibility/v01/for wc/same_ok'
python_txt_file = '/HDD/etc/repeatablility/1st/deeplabv3plus_2/filenames/IoU-0.05_Area-20_Conf-0_no_diff_no_points.txt'
python_dir = '/HDD/etc/repeatablility/1st/deeplabv3plus_2/iou0.05_area20/no_diff_w_points'
output_dir =  '/HDD/etc/repeatablility/1st/talos'
# talos_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/talos result/same_ok'
# python_txt_file = '/HDD/etc/repeatablility/2nd/deeplabv3plus_2/filenames/IoU-0.05_Area-20_Conf-0_no_diff_no_points.txt'
# python_dir = '/HDD/etc/repeatablility/2nd/deeplabv3plus_2/iou0.05_area20/no_diff_w_points'
# output_dir =  '/HDD/etc/repeatablility/2nd/talos'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

talos_img_files = glob(osp.join(talos_dir, '*.jpg'))

talos_ids = []
for talos_img_file in tqdm(talos_img_files):
    filename = osp.split(osp.splitext(talos_img_file)[0])[-1]
    
    talos_ids.append(filename)
        
python_ids = []
python_txt = open(python_txt_file, 'r')
# while True: 
#     line = python_txt.readline()
#     if line is None: break
    
#     python_ids.append(line.replace('\n', ''))
    
with open(python_txt_file, "r", encoding="utf-8") as file:
    python_ids = [line.strip() for line in file]

talos_ids = set(talos_ids)
python_ids = set(python_ids)

diff_from_talos = talos_ids - python_ids 
diff_from_python = python_ids - talos_ids

print("talos: ", len(talos_ids))
print("python: ", len(python_ids))
print("Different from talos: ", len(talos_ids.difference(python_ids)))
print("Different from python: ", len(python_ids.difference(talos_ids)))
print("Same: ", len(talos_ids.intersection(python_ids)))

diff_from_talos_dir = osp.join(output_dir, 'diff_from_talos')
if not osp.exists(diff_from_talos_dir):
    os.mkdir(diff_from_talos_dir)

txt = open(osp.join(output_dir, 'diff_from_talos.txt'), 'w')
for diff_from_talos_id in diff_from_talos:
    txt.write(diff_from_talos_id + '\n')
    copyfile(osp.join(python_dir, diff_from_talos_id + '.jpg'), osp.join(diff_from_talos_dir, diff_from_talos_id + '_python.jpg'))
    talos_img_file = find_jpg_file(talos_dir, diff_from_talos_id + '.jpg')
    assert talos_img_file is not None, ValueError(f"There is no such image file: {osp.join(talos_dir, diff_from_talos_id + '.jpg')}")
    copyfile(talos_img_file, osp.join(diff_from_talos_dir, diff_from_talos_id + '_talos.jpg'))
txt.close()

diff_from_python_dir = osp.join(output_dir, 'diff_from_python')
if not osp.exists(diff_from_python_dir):
    os.mkdir(diff_from_python_dir)

txt = open(osp.join(output_dir, 'diff_from_python.txt'), 'w')
for diff_from_python_id in diff_from_python:
    txt.write(diff_from_python_id + '\n')
    talos_img_file = find_jpg_file(talos_dir + '/../diff', diff_from_python_id + '.jpg')
    assert talos_img_file is not None, ValueError(f"There is no such image file: {osp.join(talos_dir, diff_from_python_id + '.jpg')}")
    copyfile(talos_img_file, osp.join(diff_from_python_dir, diff_from_python_id + '_talos.jpg'))

txt.close()

