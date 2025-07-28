import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_clahe(img_file, output_filename):
    img = cv2.imread(img_file)  # BGR로 읽힘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE 객체 생성 (clipLimit와 tileGridSize는 조절 가능)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("CLAHE")
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(cv2.absdiff(enhanced, gray), cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)


def apply_gaussian_noise(img_file, output_filename):
    img = cv2.imread(img_file)  # BGR로 읽힘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    noise = np.random.normal(0, 5, gray.shape).astype(np.uint8)
    noisy_img = cv2.add(gray, noise)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Gaussian Noise")
    plt.imshow(noisy_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)

def apply_gaussian_blur(img_file, output_filename):
    img = cv2.imread(img_file)  # BGR로 읽힘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    highpass = cv2.subtract(gray, blur)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Gaussian Blur")
    plt.imshow(highpass, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    
def apply_edge_filter(img_file, output_filename):
    img = cv2.imread(img_file)  # BGR로 읽힘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Laplacian Edge")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
if __name__ == '__main__':
    import os.path as osp 
    
    output_dir = '/HDD/etc/outputs'
    img_file = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/repeatability/1/125032816405615_5/1_image.bmp'
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]

    apply_clahe(img_file, osp.join(output_dir, 'clahe_' + filename + '.jpg'))
    apply_gaussian_noise(img_file, osp.join(output_dir, 'gaussian_noise_' + filename + '.jpg'))
    apply_gaussian_blur(img_file, osp.join(output_dir, 'gaussian_blur_' + filename + '.jpg'))
    apply_edge_filter(img_file, osp.join(output_dir, 'edge_filter_' + filename + '.jpg'))
    

