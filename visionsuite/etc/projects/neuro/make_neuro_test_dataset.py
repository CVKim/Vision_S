import os.path as osp
import os 
import shutil 
from pathlib import Path
from tqdm import tqdm
from glob import glob 
import json
import cv2


def make_repeatability_test_dataset(input_dir, output_dir, 
                                    cases, roi, category):
    
    output_dir = osp.join(output_dir, category)
    os.makedirs(output_dir, exist_ok=True)
    
    for case in cases:
        _output_dir = osp.join(output_dir, str(case))
        os.makedirs(_output_dir, exist_ok=True)
    
        img_files = glob(osp.join(input_dir, category, str(case), "**/**.bmp"))
        for img_file in tqdm(img_files, desc=case):
            filename = osp.splitext(img_file)[0].split("/")[-2]
            img = cv2.imread(img_file)
            
            img = img[roi[1]:roi[3], roi[0]:roi[2]]
            
            cv2.imwrite(osp.join(_output_dir, filename + '.bmp'), img)
            
        

if __name__ == '__main__':                
    
    ### 1st
    input_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/1st/data'
    output_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/1st/neuro'
    cases = ['1', '2', '3']
    # category = 'repeated_ng'
    category = 'repeated_ok'

    
    # ### 2nd
    # input_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/2nd/data'
    # output_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/2nd/neuro'
    # cases = ['1', '2', '3']
    # category = 'repeated_ng'
    # # category = 'repeated_ok'
    
    roi = [220, 60, 1340, 828]
    make_repeatability_test_dataset(input_dir, output_dir, cases, roi, category)