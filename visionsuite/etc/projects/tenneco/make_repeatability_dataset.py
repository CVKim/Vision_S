import os.path as osp 
import os 
from glob import glob 
from tqdm import tqdm
import shutil

img_dirs = ['/HDD/datasets/projects/Tenneco/Metalbearing/outer/repeatability']
cases = ['1', '2', '3']
input_dirs = ['/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeat_unit/images']
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeat_unit'
folders = ['']

# img_dirs = ['/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data',
#             '/HDD/datasets/projects/Tenneco/Metalbearing/outer/repeatability'
#         ]
# cases = ['OUTER_shot01', 'OUTER_shot02', 'OUTER_shot03', '1', '2', '3']

# input_dirs = [
#             # '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/talos result/diff',
#             # '/DeepLearning/research/data/benchmarks/benchmarks_production/tenneco/repeatibility/v01/for wc/diff'
#             '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/for wc',
#             '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/talos result'
#             ]
# output_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/benchmark/repeatability'
if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
# folders = ['딥러닝 체크', '얼라인', '경계성', '순서 틀어짐', '오염', 'same_ok']
for input_dir in input_dirs:
    for folder in folders:
        
        if not osp.exists(osp.join(input_dir, folder)):
            continue

        _output_dir = osp.join(output_dir, 'datasets')
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)

        txt = open(osp.join(_output_dir, 'filenames.txt'), 'a')
        img_files  = glob(osp.join(input_dir, folder, '*.jpg'))
        
        for img_file in tqdm(img_files, desc=folder):
            filename = osp.split(osp.splitext(img_file)[0])[-1]
            txt.write(filename + '\n')

            for img_dir in img_dirs:
                for case in cases:
                    
                    if osp.exists(osp.join(img_dir, case)):
                        if osp.exists(osp.join(img_dir, case, filename)):
                            
                            try:
                                if 'Outer' in filename:
                                    _filename = filename.replace('_Outer', "")
                                else:
                                    _filename = filename
                                if not osp.exists(osp.join(_output_dir, case[-1])):
                                    os.mkdir(osp.join(_output_dir, case[-1]))
                                
                                if 'shot' in case:
                                    _case = case[-1]
                                else:
                                    _case = case
                                
                                if osp.exists(osp.join(_output_dir, _case, _filename)):
                                    continue

                                shutil.copytree(osp.join(img_dir, case, filename), osp.join(_output_dir, _case, _filename))     
                            except Exception as error:
                                print(error)
                                                                
