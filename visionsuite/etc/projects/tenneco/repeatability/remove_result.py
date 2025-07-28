import json 
import shutil

# first = '/HDD/etc/repeatablility/talos2/1st/benchmark/deeplabv3plus/outputs/outputs.json'
# second = '/HDD/etc/repeatablility/talos3/1st/benchmark/segformer_b2_unfrozen_w1120_h768_tta/outputs/outputs.json'
# dir_name = '/HDD/etc/repeatablility/talos3/1st/benchmark/segformer_b2_unfrozen_w1120_h768_tta'
first = '/HDD/etc/repeatablility/talos2/2nd/benchmark/deeplabv3plus/outputs/outputs.json'
second = '/HDD/etc/repeatablility/talos3/2nd/benchmark/segformer_b2_unfrozen_w1120_h768_tta/outputs/outputs.json'
dir_name = '/HDD/etc/repeatablility/talos3/2nd/benchmark/segformer_b2_unfrozen_w1120_h768_tta'

with open(first, 'r') as jf:
    first = json.load(jf)
    
with open(second, 'r') as jf:
    second = json.load(jf)
    
first_ids = []
for key in ['오염', '딥러닝', '경계성', 'repeated_ng', 'repeated_ok']:
    if key in first:
        first_ids += first[key]['repeated']['ok_id']
        first_ids += first[key]['repeated']['ng_id']
        first_ids += first[key]['not_repeated']['id']
        
assert len(first_ids) == first['total_count']


second_ids = []
for key in ['오염', '시인성', '딥러닝 바보', '한도 경계성', '종횡비 경계성', '기타 불량', 'repeated_ok']:
    if key in second:
        second_ids += second[key]['repeated']['ok_id']
        second_ids += second[key]['repeated']['ng_id']
        second_ids += second[key]['not_repeated']['id']
    
assert len(second_ids) == second['total_count']


only_first, only_second = [], []
for val in first_ids:
    if val in second_ids:
        continue 
    else:
        only_first.append(val)
        
for val in second_ids:
    if val in first_ids:
        continue 
    else:
        only_second.append(val)
        
        
print(len(only_first))
print(len(only_second))
# print(only_second)

assert len(only_second) == second['total_count'] - first['total_count']

exception_ids = []
for key in only_second:
    if key in second['repeated_ok']['repeated']['ng_id'] or key in second['repeated_ok']['repeated']['ok_id'] or key in second['repeated_ok']['not_repeated']['id']:
        continue
    else:
        exception_ids.append(key)
        
import os.path as osp 
import os 
from tqdm import tqdm 

for key in tqdm(only_second):
    for order in range(1, 4):
        for fov in range(1, 15):
            
            if order == 1:
                json_file = osp.join(dir_name, 'repeated_ok', f'exp/labels', f'{key}_{fov}_1_image.json')
            else:
                json_file = osp.join(dir_name, 'repeated_ok', f'exp{order}/labels', f'{key}_{fov}_1_image.json')
                
            assert osp.exists(json_file)
            
            os.remove(json_file)
