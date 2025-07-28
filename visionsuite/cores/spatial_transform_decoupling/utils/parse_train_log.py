import json
import os.path as osp

def parse_train_log(log_filename):
    
    assert osp.exists(log_filename), ValueError(f'There is no such log-file: {log_filename}')
    
    json_data = []
    with open(log_filename, 'r') as f:
        for line in f:
            json_data.append(json.loads(line))
    
    
    
if __name__ == '__main__':
    log_filename = '/HDD/datasets/projects/rich/24.06.19/benchmark/std/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_dota_ms_rr_le90_stdc_xyawh321v/20240902_055256.log.json'
    parse_train_log(log_filename)