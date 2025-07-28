import os 
import os.path as osp 
from glob import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm

def vis_raw_output_by_channel(trt_arr, python_arr, _output_dir, filename):
    ### Vis. raw output 
    channels = trt_arr.shape[-1]
    vmin = min(trt_arr.min(), python_arr.min())
    vmax = max(trt_arr.max(), python_arr.max())

    fig, axes = plt.subplots(2, channels, figsize=(5 * channels, 10))
    mappable = None

    for c in range(channels):
        ax = axes[0, c]
        im = ax.imshow(trt_arr[..., c], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"TALOS (Channel {c})")
        ax.axis("off")
        if mappable is None:
            mappable = im

    for c in range(channels):
        ax = axes[1, c]
        ax.imshow(python_arr[..., c], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"Python (Channel {c})")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # 오른쪽에 세로로 길게
    fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, cax=cbar_ax)

    plt.suptitle("Raw Prediction per Channel", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.90, 0.95])  # [left, bottom, right, top]
    plt.savefig(osp.join(_output_dir, filename + '_raw.png'))
    plt.close()

def vis_argmax_output(trt_arr, python_arr, _output_dir, filename):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    argmax_trt_arr = np.argmax(trt_arr, axis=-1)*255
    argmax_python_arr = np.argmax(python_arr, axis=-1)*255
    
    ax = axes[0]
    ax.imshow(argmax_trt_arr)
    ax.set_title(f"TALOS")
    ax.axis("off")
    
    ax = axes[1]
    ax.imshow(argmax_python_arr)
    ax.set_title(f"PYTHON")
    ax.axis("off")
    
    plt.suptitle("Argmax Prediction", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0., 0.95])  # [left, bottom, right, top]
    plt.savefig(osp.join(_output_dir, filename + '_argamx.png'))
    plt.close()

def vis_pred(model, npz_file, trt_input_dir, python_input_dir, _output_dir, filename):
    # pred images ------------------------------------------------------------
    trt_pred_img = cv2.imread(osp.join(trt_input_dir, filename + '.bmp.bmp.png'))
    if trt_pred_img is None:
        trt_pred_img = cv2.imread(osp.join(trt_input_dir, filename + '.bmp.bmp'))
    if model == 'deeplabv3plus':
        python_pred_img = cv2.imread(osp.join(python_input_dir, '../' + filename + '_argmax.png'))
    else:
        python_pred_img = cv2.imread(osp.join(python_input_dir, '../vis/' + osp.split(osp.splitext(npz_file)[0])[-1] + '.png'))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(trt_pred_img, cmap='gray')
    axes[0].axis("off")

    axes[1].imshow(python_pred_img, cmap='gray')
    axes[1].axis("off")

    plt.suptitle(f"trt (left) vs. python (right)", fontsize=14)
    plt.tight_layout()
    plt.savefig(osp.join(_output_dir, filename + '_pred.png'))
    plt.close()
    
def vis_historgram_by_channel(trt_arr, python_arr, is_same, rtol, atol, ratio_threshold, _output_dir, filename):
    channels = trt_arr.shape[-1]
    # --- 히스토그램 ---
    # First histogram
    fig = plt.figure(figsize=(30, 10))
    plt.subplots_adjust(bottom=0.2)  # 하단 여백을 충분히 줌
    fig.text(0.5, 0.04, f"same = {is_same}",  ha='center', fontsize=10)
    fig.text(0.5, 0.01, f"(rtol = {rtol}, atol = {atol}, ratio_threshold = {ratio_threshold})%", ha='center', fontsize=10)
    for channel in range(channels):
        plt.subplot(2, channels, channel + 1)
        n, bins, patches = plt.hist(trt_arr[..., channel].flatten(), bins=100, color='skyblue', edgecolor='k')
        plt.title(f"TALOS channel {channel}")
        plt.xlabel("Value")
        plt.ylabel("Pixel Count(log)")
        plt.grid(True)
        plt.yscale("log")  # 로그 스케일

        # # 막대 위에 개수 표시
        # for count, bin_left, patch in zip(n, bins, patches):
        #     if count > 0:
        #         plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
        #                 ha='center', va='bottom', fontsize=6, rotation=90)

    # Second histogram
    for channel in range(channels):
        plt.subplot(2, channels, channel + 1 + channels)
        n2, bins2, patches2 = plt.hist(python_arr[..., channel].flatten(), bins=100, color='skyblue', edgecolor='k')
        plt.title(f"Python channel {channel}")
        plt.xlabel("Value")
        plt.ylabel("Pixel Count(log)")
        plt.grid(True)
        plt.yscale("log")  # 로그 스케일
        
        # for count, bin_left, patch in zip(n2, bins2, patches2):
        #     if count > 0:
        #         plt.text(bin_left + (bins2[1] - bins2[0]) / 2, count, f"{int(count)}", 
        #                 ha='center', va='bottom', fontsize=6, rotation=90)

    # plt.tight_layout()
    plt.savefig(osp.join(_output_dir, filename + '_histo.jpg'))
    plt.close()


def main():
    # orders = ['1st', '2nd']
    orders = ['1st']
    cases = ['diff_from_python', 'diff_from_talos']
    # model = 'mask2former'
    model = 'deeplabv3plus'

    height, width, channel = 768, 1120, 4
    rtol = 1e-2
    atol = 1e-2
    ratio_threshold = 0 # %

    for order in orders:
        for case in cases:
            num_same, num_not_same = 0, 0
            trt_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/talos/{model}/{order}/{case}'       
            if model == 'deeplabv3plus':
                python_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test/{order}/{case}/vis/raw'
            else:
                python_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test/{order}/{case}/raw'
            output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/compare/{model}/ratio{ratio_threshold}_rtol{rtol}_atol{atol}/{order}/{case}'

            if not osp.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            bin_files = glob(osp.join(trt_input_dir, '*.bin'))
            # bin_files = glob(osp.join(trt_input_dir, '125020717055621_14_Outer_1.bmp.bmp.bin'))
            if model != 'deeplabv3plus':
                npz_files = glob(osp.join(python_input_dir, '*.npy'))

            for bin_file in tqdm(bin_files, desc=f"order-{order}-case-{case}"):
                filename = osp.split(osp.splitext(bin_file)[0])[-1].split(".")[0]
                if model == 'deeplabv3plus':
                    npz_file = osp.join(python_input_dir, filename + '.npz')
                else:
                    _filename = [f for f in npz_files if osp.split(osp.splitext(f)[0])[-1].startswith(filename)]                    
                    assert len(_filename) == 1, ValueError(f"There are more than 2 filenames: {_filename}")
                    npz_file = _filename[0]
                
                if not osp.exists(npz_file):
                    print(f'There is no such npz file: {npz_file}')
                    continue
                
                
                trt_arr = np.fromfile(bin_file, dtype=np.float32).reshape((channel, height, width))
                trt_arr = np.transpose(trt_arr, (1, 2, 0))
                python_arr = np.load(npz_file)
                if 'arr' in python_arr:
                    python_arr = np.load(npz_file)['arr']
                if python_arr.shape == (channel, height, width):
                    python_arr = np.transpose(python_arr, (1, 2, 0))
                    
                
                assert trt_arr.shape == python_arr.shape # hwc
                assert trt_arr.dtype == python_arr.dtype # float32    
                        
                is_close = np.allclose(trt_arr, python_arr, rtol=rtol, atol=atol)
                is_same = is_close

                diff = np.abs(trt_arr - python_arr)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                ratios = []
                for ch in range(trt_arr.shape[-1]):
                    ratio = np.sum(np.abs(trt_arr[..., ch] - python_arr[..., ch]) > rtol)/diff.size*100
                    ratios.append(ratio)
                    
                    if not is_close and ratio > ratio_threshold:
                        is_same = False
                    
                _output_dir = osp.join(output_dir, str(is_same))
                if not osp.exists(_output_dir):
                    os.mkdir(_output_dir)
                
                vis_raw_output_by_channel(trt_arr, python_arr, _output_dir, filename)
                vis_argmax_output(trt_arr, python_arr, _output_dir, filename)
        
                vis_pred(model, npz_file, trt_input_dir, python_input_dir, _output_dir, filename)
                vis_historgram_by_channel(trt_arr, python_arr, is_same, rtol, atol, ratio_threshold, _output_dir, filename)

            num_txt = open(osp.join(output_dir, 'numbers.txt'), 'w')
            num_txt.write(f'number of same: {num_same}\n')
            num_txt.write(f'number of NOT same: {num_not_same}\n')
            num_txt.close()


if __name__ == '__main__':
    main()