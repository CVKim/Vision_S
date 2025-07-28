import cv2 
import os.path as osp
import os 

input_dir = '/HDD/etc/repeatablility/talos2/2nd/benchmark/mask2former_swin-s_w1120_h768/outputs/check_shift/125032816412602'
img_filename = '2_6'
output_dir = '/HDD/etc/repeatablility/talos2/2nd/benchmark/mask2former_swin-s_w1120_h768/outputs/check_shift/125032816412602/test_2'
roi = [220, 60, 1340, 828]

if not osp.exists(output_dir):
    os.mkdir(output_dir)

img_file = osp.join(input_dir, img_filename + '.bmp')
img = cv2.imread(img_file)

# roi
_img = img[roi[1]:roi[3], roi[0]:roi[2]]
cv2.imwrite(osp.join(output_dir, img_filename + '_roi.bmp'), _img)

# roi + translate
offset_xs = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20]
offset_ys = [-10, -5, 0, 5, 10]

for offset_x in offset_xs:
    for offset_y in offset_ys:
        _img = img[roi[1] + offset_y:roi[3] + offset_y, roi[0] + offset_x:roi[2] + offset_x]
        cv2.imwrite(osp.join(output_dir, img_filename + f'_roi_x{offset_x}_y{offset_y}.bmp'), _img)




