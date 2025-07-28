import os 
import os.path as osp 
import glob 
from shutil import copyfile
import json
from visionsuite.utils.helpers import get_filename
import random
from tqdm import tqdm


def split_dataset(input_dir, output_dir, image_formats, ratio, figs=True):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    img_files = [file for image_format in image_formats for file in glob.glob(osp.join(input_dir, f'*/*.{image_format}'))]

    label2idx = {}
    label2cnt, train_label2cnt, val_label2cnt = {}, {}, {}
    for img_file in tqdm(img_files):
        
        json_file = osp.splitext(img_file)[0] + '.json'
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        labels = []
        for ann in anns:
            label = ann['label']
            if label not in label2idx:
                label2idx[label] = len(label2idx)
                label2cnt[label], train_label2cnt[label], val_label2cnt[label] = 0, 0, 0
                
            label2cnt[label] += 1
            labels.append(label)
            
        if random.uniform(0, 1) <= ratio:
            mode = 'val'
        else:
            mode = 'train'
            
        for label in labels:
            if int(train_label2cnt[label]*(ratio/(1 - ratio))) > val_label2cnt[label]:
                mode = 'val'
                break
            
        if mode == 'train':
            for label in labels:
                train_label2cnt[label] += 1
        elif mode == 'val':
            for label in labels:
                val_label2cnt[label] += 1
        

        _output_dir = osp.join(output_dir, mode)
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
            
        copyfile(img_file, osp.join(_output_dir, get_filename(img_file, True)))            
        copyfile(json_file, osp.join(_output_dir, get_filename(json_file, True)))            
        
        
    f = open(osp.join(output_dir, 'classes.txt'), 'w')
    for key, val in label2idx.items():
        f.write(f"{val}: {key}\n")        
    f.close()

    if figs:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(train_label2cnt.keys()),
                y=list(train_label2cnt.values()),
                name="Train",
                marker_color="blue",
                text=list(train_label2cnt.values()),  # Add text for Train bars
                textposition='auto'  # Position text automatically (usually above the bar)
            )
        )

        fig.add_trace(
            go.Bar(
                x=list(val_label2cnt.keys()),
                y=list(val_label2cnt.values()),
                name="Val",
                marker_color="red",
                text=list(val_label2cnt.values()),  # Add text for Val bars
                textposition='auto'  # Position text automatically
            )
        )
        fig.write_image(osp.join(output_dir, "split.png"), engine="auto")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(label2cnt.keys()),
                y=list(label2cnt.values()),
                name="Total",
                marker_color="green",
                text=list(label2cnt.values()),  # Add text for Total bars
                textposition='auto'  # Position text automatically
            )
        )
        fig.write_image(osp.join(output_dir, "total.png"), engine="auto")

                
if __name__ == '__main__':
    input_dir = '/DeepLearning/_athena_tests/datasets/rectangle2/raw_dataset'
    output_dir = '/DeepLearning/_athena_tests/datasets/rectangle2/split_dataset'
    image_formats = ['bmp']
    ratio = 0.3
    figs = True
    
    split_dataset(input_dir, output_dir, image_formats, ratio, figs=True)
        
            
            
            
            
            
            
            
            
            
            
        
        
        