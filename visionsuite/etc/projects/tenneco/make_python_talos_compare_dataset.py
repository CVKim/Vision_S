import os.path as osp 
from glob import glob 
import os
import shutil
from tqdm import tqdm 

input_dirs = ['/DeepLearning/etc/_athena_tests/benchmark/talos/1st',
              '/DeepLearning/etc/_athena_tests/benchmark/talos/2nd'
]

img_dirs = ['/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data',
            '/HDD/datasets/projects/Tenneco/Metalbearing/outer/repeatability'
        ]
cases = ['OUTER_shot01', 'OUTER_shot02', 'OUTER_shot03', '1', '2', '3']

for input_dir in input_dirs:
    txt_files = glob(osp.join(input_dir, '*.txt'))
    
    for txt_file in tqdm(txt_files, desc=input_dir):
        txt_filename = osp.split(osp.splitext(txt_file)[0])[-1]
        output_dir = osp.join(input_dir, f'images_{txt_filename}')
        
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
                                 
        txt = open(txt_file, 'r')
        while True:
            line = txt.readline()
            
            if not line: break 
            
            filename = line.replace('\n', '')
            
            for img_dir in img_dirs:
                idx = 0
                for case in cases:
                    dir_name = osp.join(img_dir, case, filename)
                    
                    if not osp.exists(dir_name):
                        continue 
                    else:
                        img_file = glob(osp.join(dir_name, '*.bmp'))
                        assert len(img_file) == 1, ValueError(f"There are more than 1 bmp")
                        img_file = img_file[0]
                        img_filename = osp.split(osp.splitext(img_file)[0])[-1]
                        shutil.copyfile(img_file, osp.join(output_dir, filename + f'_{idx}.bmp'))     
                        idx += 1
                        
                        
                    
                    
            
            
            
            
              