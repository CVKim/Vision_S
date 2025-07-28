import os 
import os.path as osp 
from glob import glob 
from shutil import copyfile
from tqdm import tqdm

# mode = '1st'
mode = '2nd'

input_dir = f'/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeatability_check_python/{mode}/deeplabv3plus/outputs/not_repeated/vis/재분류'
cats = os.listdir(input_dir)

cat2filename = {}
for cat in cats:
    cat2filename[cat] = []
    img_files = glob(osp.join(input_dir, cat, '*.bmp'))
    for img_file in img_files:
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        cat2filename[cat].append(filename)
    
print(cat2filename)


candi_input_dir = f'/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeatability_check_python/{mode}/segformer_b2_unfrozen_w1120_h768_tta/outputs/not_repeated/vis'
cats = os.listdir(candi_input_dir)

candi_cat2filename = {}
for cat in cats:
    candi_cat2filename[cat] = []
    img_files = glob(osp.join(candi_input_dir, cat, '*.bmp'))
    for img_file in img_files:
        candi_cat2filename[cat].append(img_file)
    
for candi_cat, candi_filenames in candi_cat2filename.items():
    for candi_filename in tqdm(candi_filenames, desc=candi_cat):
        filename = osp.split(osp.splitext(candi_filename)[0])[-1]
        for target_cat, target_filenames in cat2filename.items():
            if filename in target_filenames:
                output_dir = osp.join(candi_input_dir, '재분류', target_cat)
                os.makedirs(output_dir, exist_ok=True)
                
                copyfile(candi_filename, osp.join(output_dir, filename + '.bmp'))
                print(f"ADDED at {target_cat}")
                
            
                