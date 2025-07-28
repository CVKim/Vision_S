import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm


mode = 'val'
output_dir = f"/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/figs/{mode}"
candidates = ['m2f_epochs100', 'cosnet_epochs100', 'pidnet_epochs100']


if not osp.exists(output_dir):
    os.mkdir(output_dir)

candidates_info = {}
metrics = []
for candidate in candidates:
    df = pd.read_csv(f"/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{candidate}/train/logs/{mode}/{mode}.csv")
    candidates_info.update({candidate: df})
    metrics += list(df.columns.values)
    
metrics = set(metrics)
    
for metric in tqdm(metrics):

    plt.figure(figsize=(20, 10))
    for key, val in candidates_info.items():
        plt.plot(val[metric], label=key, linewidth=1.5)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend()
    plt.suptitle(f'{metric}')
    plt.savefig(osp.join(output_dir, f'{metric}.png'))
    plt.close()

    
if mode == 'train':
    tf_df = pd.read_csv(f"/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/tf_deeplabv3plus_epochs100/train/logs/{mode}/{mode}_epoch.csv")
    tf2mmseg_metrics = {'loss': ['loss', 1], 'gpu_usage': ['GPU 0 Mem. (GB)', 1], 'time_for_a_epoch': ['duration (sec)', 1], 
                        'learning_rate': ['lr_0', 1], 'global_acc': ['aAcc', 100], 
                        'CHAMFER_MARK_iou': ['IoU_CHAMFER_MARK', 100], 'MARK_iou': ['IoU_MARK', 100], 'LINE_iou': ['IoU_LINE', 100], 
                        'CHAMFER_MARK_acc': ['Acc_CHAMFER_MARK', 100], 'MARK_acc': ['Acc_MARK', 100], 'LINE_acc': ['Acc_LINE', 100]}
elif mode == 'val':
    tf_df = pd.read_csv(f"/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/tf_deeplabv3plus_epochs100/train/logs/{mode}/{mode}_1.csv")
    tf2mmseg_metrics = {'gpu_usage': ['GPU 0 Mem. (GB)', 1], 'time_for_a_epoch': ['duration (sec)', 1], 
                        'global_acc': ['aAcc', 100], 
                        'CHAMFER_MARK_iou': ['IoU_CHAMFER_MARK', 100], 'MARK_iou': ['IoU_MARK', 100], 'LINE_iou': ['IoU_LINE', 100], 
                        'CHAMFER_MARK_acc': ['Acc_CHAMFER_MARK', 100], 'MARK_acc': ['Acc_MARK', 100], 'LINE_acc': ['Acc_LINE', 100]}
    
for tf_key, mmseg_key in tqdm(tf2mmseg_metrics.items()):

    plt.figure(figsize=(20, 10))
    plt.plot(tf_df[tf_key]*mmseg_key[1], label='tf_deeplabv3plus-effb3', linewidth=1.5)
    for key, val in candidates_info.items():
        plt.plot(val[mmseg_key[0]], label=key, linewidth=1.5)
    plt.ylabel(mmseg_key[0])
    plt.xlabel('epoch')
    plt.legend()
    plt.suptitle(f'{mmseg_key[0]}')
    plt.savefig(osp.join(output_dir, f'{mmseg_key[0]}.png'))
    plt.close()




