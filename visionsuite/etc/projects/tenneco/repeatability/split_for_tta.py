import os
import os.path as osp 
from glob import glob 
from shutil import copytree
from tqdm import tqdm 

input_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250605/2nd/data/repeated_ok'
orders = [1, 2, 3]
count_threshold = 900

for order in orders:
    folders = os.listdir(osp.join(input_dir, str(order)))
    
    order2 = 1
    count = 0
    for folder in tqdm(folders, desc=f'{order} > {order2}'):
        output_dir = osp.join(input_dir, f'{str(order)}_{order2}')
        os.makedirs(output_dir, exist_ok=True)
        
        copytree(osp.join(input_dir, str(order), folder), osp.join(input_dir, f'{str(order)}_{order2}', folder))
        
        count += 1
        
        if count >= count_threshold:
            count = 0
            order2 += 1
    
    
    