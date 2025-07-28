from glob import glob 
import os.path as osp
import os


input_dir_1 = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/for wc/diff'
input_dir_2 = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/talos_result_wc/diff'
folders_1 = os.listdir(osp.join(input_dir_1))
folders_2 = os.listdir(osp.join(input_dir_2))

folders_1 += ['../same_ng', '../same_ok']
folders_2 += ['../same_ng', '../same_ok']

total_img_files = []
for folder_1 in folders_1:
    if folder_1 == '순서 틀어짐':
        continue
    total_img_files += glob(osp.join(input_dir_1, folder_1, '*.jpg'))
    
for folder_2 in folders_2:
    if folder_1 == '순서 틀어짐':
        continue
    total_img_files += glob(osp.join(input_dir_2, folder_2, '*.jpg'))
    
    
print(len(total_img_files))
filenames = []
for img_file in total_img_files:
    filename = osp.split(osp.splitext(img_file)[0])[-1].split("_")[0]
    
    if filename in filenames:
        continue 
    else:
        filenames.append(filename)
        
print(len(set(filenames)))