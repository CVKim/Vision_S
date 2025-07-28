import os.path as osp
from glob import glob 
import os 
from shutil import copyfile 
from tqdm import tqdm


input_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/data/1'
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/trt_vs_python/data/1'

if not osp.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
folders = os.listdir(input_dir)

for folder in tqdm(folders):
    img_file = osp.join(input_dir, folder, '1_image.bmp')
    
    copyfile(img_file, osp.join(output_dir, folder + '.bmp'))

