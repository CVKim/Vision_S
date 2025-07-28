
import os.path as osp 
import glob 
import os
import json
from shutil import copyfile
from tqdm import tqdm


def get_classes_info(input_dir):
    modes = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

    if len(modes) == 0:
        modes = ['./']
        
    classes = {}
        
    for mode in modes:
        json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
        
        classes[mode] = {}
        for json_file in tqdm(json_files, desc=mode):
            with open(json_file, 'r') as jf:
                anns = json.load(jf)['shapes']
                
            for ann in anns:
                label = ann['label']

                if label not in classes[mode]:
                    classes[mode][label] = 1
                else:
                    classes[mode][label] += 1    
                
    return classes 

def balance_dataset(input_dir, output_dir, counts, image_format):
    modes = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

    if len(modes) == 0:
        modes = ['./']
        
    if not osp.exists(output_dir):
        os.mkdir(output_dir)    
    
    for mode in modes:
        json_files_to_move = []
        json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
        
        _output_dir = osp.join(output_dir, mode)
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)    

        for json_file in tqdm(json_files, desc=mode):
            with open(json_file, 'r') as jf:
                anns = json.load(jf)['shapes']
                
            for ann in anns:
                label = ann['label']
                
                if counts[label] > 0:
                    counts[label] -= 1
                    
                    json_files_to_move.append(json_file)
                    
        for json_file_to_move in set(json_files_to_move):
            filename = osp.split(osp.splitext(json_file_to_move)[0])[-1]
            img_file = osp.splitext(json_file_to_move)[0] + f".{image_format}"
            copyfile(img_file, osp.join(_output_dir, filename + f".{image_format}"))
            copyfile(json_file_to_move, osp.join(_output_dir, filename + ".json"))             
    
    
if __name__ == '__main__':
    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/inner/split_dataset'
    output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/inner/split_dataset_balanced'
    
    classes_info = get_classes_info(input_dir)
    
    print(classes_info)
    
    counts = {'STABBED': 100, 'STABBED_C': 100, 'SCRATCH': 100, 'STABBED_P': 55}   
    image_format = 'bmp' 

    balance_dataset(input_dir, output_dir, counts, image_format)
    
    
    classes_info = get_classes_info(output_dir)
    print(classes_info)
