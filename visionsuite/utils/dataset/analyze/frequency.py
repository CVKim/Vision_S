import cv2
import numpy as np
import matplotlib.pyplot as plt



def check_laplaican(img_file, output_filename):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # Laplacian 연산
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap_abs = np.abs(lap)

    # 시각화
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.xticks(np.linspace(0, w, num=5, dtype=int))
    plt.yticks(np.linspace(0, h, num=5, dtype=int))

    plt.subplot(1, 2, 2)
    plt.title("Laplacian (High-Frequency Map)")
    plt.imshow(lap_abs, cmap='gray')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.xticks(np.linspace(0, w, num=5, dtype=int))
    plt.yticks(np.linspace(0, h, num=5, dtype=int))

    plt.tight_layout()
    plt.savefig(output_filename)
    
def check_fft(img_file, output_filename):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

    # 좌표 중심값
    cx, cy = w // 2, h // 2

    # 시각화
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.xticks(np.linspace(0, w, num=5, dtype=int))
    plt.yticks(np.linspace(0, h, num=5, dtype=int))

    plt.subplot(1, 2, 2)
    plt.title("FFT Magnitude Spectrum (Frequency Domain)")
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.xlabel("Frequency (Width direction)")
    plt.ylabel("Frequency (Height direction)")
    plt.xticks([0, cx, w-1], labels=["-f_x", "0", "+f_x"])
    plt.yticks([0, cy, h-1], labels=["-f_y", "0", "+f_y"])

    plt.tight_layout()
    plt.savefig(output_filename)

    
if __name__ == '__main__':
    import os.path as osp 
    
    output_dir = '/HDD/etc/outputs'
    img_file = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/repeatability/1/125032816405615_5/1_image.bmp'
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]

    check_laplaican(img_file, osp.join(output_dir, 'laplacian_' + filename + '.jpg'))
    check_fft(img_file, osp.join(output_dir, 'fft_' + filename + '.jpg'))
