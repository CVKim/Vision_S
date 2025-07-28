# case = '1st'
case = '2nd'

### height info
import pandas as pd
height_info_csv = '/HDD/etc/repeatablility/250616_OD_ROI_반복성테스트.xlsx'
output_dir = f'/HDD/etc/repeatablility/talos3/{case}/benchmark/deeplabv3plus_w1120_h768'
if case == '1st':
    sheet_name = f'repeat_250605_1st(th)' # 'repeat_250605_2nd(th)', 'repeat_250605_2nd(aspect)'
    height_info_df = pd.read_excel(height_info_csv, sheet_name=sheet_name)

    height_info = {}
    
    
    for order in [1, 2, 3]:
        height_info[order] = {}            
        order_df = height_info_df[height_info_df['repeat'] == order]
        for id, height in zip(order_df['innerid'], order_df['height']):
            inner_id = id[1:].split("_")[0].rstrip().lstrip()
            fov = id.split("_")[1].rstrip().lstrip()
            if inner_id not in height_info[order]:
                height_info[order][inner_id] = {fov: height}
            else:
                height_info[order][inner_id][fov] = height
    
elif case == '2nd':
    sheet_name = ['repeat_250605_2nd(th)', 'repeat_250605_2nd(aspect)']
    
    height_info_df = pd.read_excel(height_info_csv, sheet_name=sheet_name)
    height_info = {}
    
    for key, val in height_info_df.items():  
        for order in [1, 2, 3]:
            if order not in height_info:
                height_info[order] = {}                
            order_df = val[val['repeat'] == order]
            for id, height in zip(order_df['innerid'], order_df['height']):
                inner_id = id[1:].split("_")[0].rstrip().lstrip()
                fov = id.split("_")[1].rstrip().lstrip()
                if inner_id not in height_info[order]:
                    height_info[order][inner_id] = {fov: height}
                else:
                    height_info[order][inner_id][fov] = height
                

import matplotlib.pyplot as plt 
import os.path as osp
import seaborn as sns 
sns.set(style='darkgrid')


cnt = 0
plt.figure(figsize=(24, 12))
nx, ny = 2, 3
for idx, (key, val) in enumerate(height_info[1].items()):

    plt.subplot(nx, ny, idx%(nx*ny) + 1)
    second_val = height_info[2][key]
    third_val = height_info[3][key]
    heights, second_heights, third_heights = [], [], []
    for fov in [14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        heights.append(val[str(fov)])
        second_heights.append(second_val[str(fov)])
        third_heights.append(third_val[str(fov)])
        
    plt.plot(range(1, 15), heights, label='1st')
    plt.plot(range(1, 15), second_heights, label='2nd')
    plt.plot(range(1, 15), third_heights, label='3rd')
    plt.title(key)
    plt.xlabel('fov')
    plt.ylabel('height')
    plt.legend()
    
    if idx%(nx*ny) == 0 and idx != 0:
        plt.savefig(osp.join(output_dir, f'height_{cnt}.png'), dpi=600)
        cnt += 1
        plt.close()
        plt.figure(figsize=(24, 12))
    
    
    

