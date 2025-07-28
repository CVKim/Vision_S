import os 
import os.path as osp 
from glob import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
from shapely.geometry import Polygon
import pandas as pd

def vis_raw_output_by_channel(trt_arr, python_arr, _output_dir, filename):
    ### Vis. raw output 
    channels = trt_arr.shape[-1]
    vmin = min(trt_arr.min(), python_arr.min())
    vmax = max(trt_arr.max(), python_arr.max())

    fig, axes = plt.subplots(3, channels, figsize=(5 * channels, 10))
    mappable = None

    for c in range(channels):
        ax = axes[0, c]
        im = ax.imshow(trt_arr[..., c], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"TRT (Channel {c})")
        ax.axis("off")
        if mappable is None:
            mappable = im

    for c in range(channels):
        ax = axes[1, c]
        ax.imshow(python_arr[..., c], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"Python (Channel {c})")
        ax.axis("off")

    for c in range(channels):
        ax = axes[2, c]
        ax.imshow(abs(python_arr[..., c] - trt_arr[..., c]), cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"ABS(Python - TRT) (Channel {c})")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # 오른쪽에 세로로 길게
    fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, cax=cbar_ax)

    plt.suptitle("Raw Prediction per Channel", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.90, 0.95])  # [left, bottom, right, top]
    plt.savefig(osp.join(_output_dir, filename + '_raw.png'))
    plt.close()


def main():
    height, width, num_classes = 768, 1120, 4
    rtol = 1e-2
    atol = 1e-2    
    
    dir_path = '/DeepLearning/etc/_athena_tests/benchmark/trt_vs_python/data_unit/1'
    
    trt_bin_file = glob(osp.join(dir_path, 'trt', '*.bin'))[0]
    trt_trtexec_bin_file = glob(osp.join(dir_path, 'trt_trtexec', '*.bin'))[0]
    trt_trtexec_fp32_precision_bin_file = glob(osp.join(dir_path, 'trt_trtexec_fp32_precision', '*.bin'))[0]

    filename = osp.split(osp.splitext(trt_bin_file)[0])[-1].split(".")[0]
    
    trt_arr = np.fromfile(trt_bin_file, dtype=np.float32).reshape((num_classes, height, width))
    trt_arr = np.transpose(trt_arr, (1, 2, 0))
    trt_trtexec_arr = np.fromfile(trt_trtexec_bin_file, dtype=np.float32).reshape((num_classes, height, width))
    trt_trtexec_arr = np.transpose(trt_trtexec_arr, (1, 2, 0))
    trt_trtexec_fp32_precision_arr = np.fromfile(trt_trtexec_fp32_precision_bin_file, dtype=np.float32).reshape((num_classes, height, width))
    trt_trtexec_fp32_precision_arr = np.transpose(trt_trtexec_fp32_precision_arr, (1, 2, 0))

    print('trt vs. trt_exec: ', np.allclose(trt_arr, trt_trtexec_arr, rtol=rtol, atol=atol))
    print('trt vs. trt_trtexec_fp32_precision: ', np.allclose(trt_arr, trt_trtexec_fp32_precision_arr, rtol=rtol, atol=atol))
    print('trt_exec vs. trt_trtexec_fp32_precision: ', np.allclose(trt_trtexec_arr, trt_trtexec_fp32_precision_arr, rtol=rtol, atol=atol))

if __name__ == '__main__':
    main()