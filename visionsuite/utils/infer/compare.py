import os 
import os.path as osp 
from glob import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm

orders = ['1st']
cases = ['diff_from_python']
model = 'deeplabv3plus'

height, width, channel = 768, 1120, 4
rtol = 1e-2
atol = 1e-2
ratio_threshold = 0 # %

for order in orders:
    for case in cases:
        num_same, num_not_same = 0, 0
        python_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test/{order}/{case}/vis/raw'
        onnx_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test_onnx/{order}/{case}/raw'       
        trt_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test_trt/{order}/{case}/raw'       
        output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/compare/ratio{ratio_threshold}_rtol{rtol}_atol{atol}/{order}/{case}'

        if not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        python_arr_files = glob(osp.join(python_input_dir, '*.npz'))

        for python_arr_file in tqdm(python_arr_files, desc=f"order-{order}-case-{case}"):
            filename = osp.split(osp.splitext(python_arr_file)[0])[-1]
            onnx_arr_file = osp.join(onnx_input_dir, filename + '.npy')
            trt_arr_file = osp.join(trt_input_dir, filename + '.npy')
            
            assert osp.exists(onnx_arr_file), ValueError(f'There is no such onnx_arr_file: {onnx_arr_file}')
            assert osp.exists(trt_arr_file), ValueError(f'There is no such trt_arr_file: {trt_arr_file}')


            python_arr = np.load(python_arr_file)['arr']
            onnx_arr = np.load(onnx_arr_file)
            trt_arr = np.load(trt_arr_file)
                
            
            assert python_arr.shape == onnx_arr.shape and  onnx_arr.shape == trt_arr.shape# hwc
            assert python_arr.dtype == onnx_arr.dtype and  onnx_arr.dtype == trt_arr.dtype# hwc
            
            is_close_1 = np.allclose(python_arr, onnx_arr, rtol=rtol, atol=atol)
            is_close_2 = np.allclose(python_arr, trt_arr, rtol=rtol, atol=atol)
            is_close_3 = np.allclose(onnx_arr, trt_arr, rtol=rtol, atol=atol)
            
            
            if is_close_1 and is_close_2 and is_close_3:
                is_same = True 
                num_same += 1
            else:
                is_same = False
                num_not_same += 1
            
            _output_dir = osp.join(output_dir, str(is_same))
            if not osp.exists(_output_dir):
                os.mkdir(_output_dir)
            
            # --- 히스토그램 ---
            # First histogram
            fig = plt.figure(figsize=(10, 20))
            plt.subplots_adjust(bottom=0.1)
            fig.text(0.5, 0.03, f"python =? onnx = {is_close_1}, python =? trt = {is_close_2}, trt =? onnx = {is_close_3}", 
                ha='center', fontsize=10)
            plt.subplot(3, 1, 1)
            n, bins, patches = plt.hist(python_arr.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("PYTHON")
            plt.xlabel("Value")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일

            # 막대 위에 개수 표시
            for count, bin_left, patch in zip(n, bins, patches):
                if count > 0:
                    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)

            # Second histogram
            plt.subplot(3, 1, 2)
            n, bins, patches = plt.hist(onnx_arr.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("ONNX")
            plt.xlabel("Value")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일

            # 막대 위에 개수 표시
            for count, bin_left, patch in zip(n, bins, patches):
                if count > 0:
                    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)

            plt.subplot(3, 1, 3)
            n, bins, patches = plt.hist(trt_arr.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("TRT")
            plt.xlabel("Value")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일

            # 막대 위에 개수 표시
            for count, bin_left, patch in zip(n, bins, patches):
                if count > 0:
                    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)

            plt.savefig(osp.join(_output_dir, filename + '_histo.jpg'))
            plt.close()
            
            
            channels = python_arr.shape[2]
            fig, axes = plt.subplots(3, channels, figsize=(5 * channels, 10))

            for c in range(channels):
                ax = axes[0, c] if channels > 1 else axes[0]
                ax.imshow(python_arr[..., c], cmap='hot', interpolation='nearest')
                ax.set_title(f"Python (Channel {c})")
                ax.axis("off")
                fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

            for c in range(channels):
                ax = axes[1, c] if channels > 1 else axes[0]
                ax.imshow(onnx_arr[..., c], cmap='hot', interpolation='nearest')
                ax.set_title(f"ONNX (Channel {c})")
                ax.axis("off")
                fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

            for c in range(channels):
                ax = axes[2, c] if channels > 1 else axes[1]
                ax.imshow(trt_arr[..., c], cmap='hot', interpolation='nearest')
                ax.set_title(f"TRT (Channel {c})")
                ax.axis("off")
                fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle("Raw Prediction per Channel", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(osp.join(_output_dir, filename + '_raw.png'))
            plt.close()
            
            
            # pred images ------------------------------------------------------------
            python_pred_img = cv2.imread(osp.join(python_input_dir, '../' + filename + '_argmax.png'))
            onnx_pred_img = cv2.imread(osp.join(onnx_input_dir, '../vis/' + filename + '.png'))
            trt_pred_img = cv2.imread(osp.join(trt_input_dir, '../vis/' + filename + '.png'))
            
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(python_pred_img, cmap='gray')
            axes[0].axis("off")

            axes[1].imshow(onnx_pred_img, cmap='gray')
            axes[1].axis("off")
            
            axes[2].imshow(trt_pred_img, cmap='gray')
            axes[2].axis("off")

            plt.suptitle(f"Python vs. Onnx vs. Trt", fontsize=14)
            plt.tight_layout()
            plt.savefig(osp.join(_output_dir, filename + '_pred.png'))
            plt.close()


        num_txt = open(osp.join(output_dir, 'numbers.txt'), 'w')
        num_txt.write(f'number of same: {num_same}\n')
        num_txt.write(f'number of NOT same: {num_not_same}\n')
        num_txt.close()

