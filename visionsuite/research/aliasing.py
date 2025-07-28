import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

# Create a high-frequency checkerboard pattern
def checkerboard(size=64, num_checks=8):
    row_check = np.kron([[1, 0] * (num_checks // 2), [0, 1] * (num_checks // 2)] * (num_checks // 2), 
                        np.ones((size // num_checks, size // num_checks)))
    return row_check

# Apply shift
def shift_image(img, shift):
    return scipy.ndimage.shift(img, shift, mode='nearest')

# Downsample (with or without blur)
def downsample(img, factor=2, anti_aliasing=False):
    if anti_aliasing:
        img = scipy.ndimage.gaussian_filter(img, sigma=1)
    return img[::factor, ::factor]

# Generate images
original = checkerboard()
shifted = shift_image(original, shift=(0, 1))  # 1 pixel to the right

# Downsample with and without anti-aliasing
original_down = downsample(original, anti_aliasing=False)
shifted_down = downsample(shifted, anti_aliasing=False)

original_blur_down = downsample(original, anti_aliasing=True)
shifted_blur_down = downsample(shifted, anti_aliasing=True)

# Plot results
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].imshow(original, cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 1].imshow(original_down, cmap='gray')
axs[0, 1].set_title("Aliased Downsample")
axs[0, 2].imshow(original_blur_down, cmap='gray')
axs[0, 2].set_title("Anti-Aliased Downsample")

axs[1, 0].imshow(shifted, cmap='gray')
axs[1, 0].set_title("Shifted (1px right)")
axs[1, 1].imshow(shifted_down, cmap='gray')
axs[1, 1].set_title("Shifted + Aliased Downsample")
axs[1, 2].imshow(shifted_blur_down, cmap='gray')
axs[1, 2].set_title("Shifted + Anti-Aliased Downsample")

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# 차이 계산
diff_aliased = np.abs(original_down - shifted_down)
diff_anti_aliased = np.abs(original_blur_down - shifted_blur_down)

# 시각화
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(diff_aliased, cmap='hot')
axs[0].set_title("Diff: Aliased Downsample")
axs[1].imshow(diff_anti_aliased, cmap='hot')
axs[1].set_title("Diff: Anti-Aliased Downsample")

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()