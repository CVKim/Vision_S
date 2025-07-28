import os.path as osp
import os 
import shutil 
from pathlib import Path
from tqdm import tqdm
from glob import glob 
import json


def make_repeatability_test_dataset(input_dir, output_dir, filenames_dir, cases, json_file, categories):
    
    with open(json_file, 'r') as jf:
        repeated = json.load(jf)
    print(f'repeated: {repeated}')
    
    for category in categories:
        _filenames_dir = osp.join(filenames_dir, category)
        filenames = glob(osp.join(_filenames_dir, "*.bmp"))
            
        for filename in tqdm(filenames, desc=f'{category}: '):
            fname = osp.split(osp.splitext(filename)[0])[-1]
            for case in cases:
                base_dir = osp.join(input_dir, str(case))
                
                for index in range(1, 15):
                    img_file = osp.join(base_dir, f'{fname}_{index}/1_image.bmp')
                    
                    assert osp.exists(img_file), ValueError(f'there is no such image file: {img_file}')
                    
                    # if 'OUTER_shot0' in case:
                    #     _case = case.replace('OUTER_shot0', '')
                    # else:
                    #     _case = case
                    
                    _output_dir = osp.join(output_dir, category, str(case), f'{fname}_{index}')
                    os.makedirs(_output_dir, exist_ok=True)
                    shutil.copyfile(img_file, osp.join(_output_dir, '1_image.bmp'))

if __name__ == '__main__':                
    # ## 1st
    # input_dir = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data'
    # filenames_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/1st/outputs/not_repeated/'
    # json_file = osp.join(filenames_dir, '../output_data.json')
    # output_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/1st/data'
    # cases = ['OUTER_shot01', 'OUTER_shot02', 'OUTER_shot03']
    # categories = ['오염', '경계성', '딥러닝', 'repeated_ng', 'repeated_ok']
    
    ### 2nd
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/data'
    filenames_dir = "/HDD/etc/repeatablility/talos2/2nd/benchmark/segformer_b2_unfrozen_w1120_h768/outputs/not_repeated/vis"
    json_file = osp.join(filenames_dir, '../../outputs.json')
    output_dir = '/HDD/etc/repeatablility/talos2/2nd/benchmark/segformer_b2_unfrozen_w1120_h768/tta'
    cases = ['1', '2', '3']
    categories = ['오염', '경계성', '딥러닝', 'repeated_ng', 'repeated_ok']
    
    make_repeatability_test_dataset(input_dir, output_dir, filenames_dir, cases, json_file, categories)