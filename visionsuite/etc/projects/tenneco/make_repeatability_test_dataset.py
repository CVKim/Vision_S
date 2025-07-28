import os.path as osp
import os 
import shutil 
from pathlib import Path
from tqdm import tqdm
from glob import glob 


def make_repeatability_test_dataset(input_dir, output_dir, filenames_dir):
    
    filenames = glob(osp.join(filenames_dir, "*.png"))
    _filenames = []
    for filename in filenames:
        filename = osp.split(osp.splitext(filename)[0])[-1]
        _filenames.append(filename)
        
    filenames = _filenames
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    cases = [1]
    for case in cases:
        base_dir = osp.join(input_dir, str(case))
        folders = sorted([Path(base_dir) / f for f in os.listdir(base_dir) if (Path(base_dir) / f).is_dir()])
        # sorted_folders = sorted(folders, key=lambda f: f.stat().st_ctime)
        print(f"There are {len(folders)} folders")
        
        _output_dir = osp.join(output_dir, str(case))
        
        if case == 1:
            sorted_folders_case_1 = folders
        
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)

        for idx, folder_dir in tqdm(enumerate(folders), desc=f"case: {str(case)}"):
            folder_dir = str(folder_dir)
            filename = folder_dir.split('/')[-1]
            
            if filename in filenames:
                img_file = osp.join(folder_dir, f"1_image.bmp")
                assert osp.exists(img_file), ValueError(f"There is no such image file: {img_file}")
                
                shutil.copyfile(img_file, osp.join(output_dir, filename + '.bmp'))
                

if __name__ == '__main__':                
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeat_unit/datasets'
    filenames_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeat_unit/segmentation/deeplabv3plus/wo_patch/no_tta/exp/미..검'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/performances/dataset'
    make_repeatability_test_dataset(input_dir, output_dir, filenames_dir)