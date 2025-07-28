from keras_unet_collection.models import unet_3plus_2d
from tqdm import tqdm
input_size = (512, 512, 3)
n_labels = 4
filter_num_down = [32, 64, 128, 256, 512]
# filter_num_skip = [32, 32, 32, 32]
# filter_num_aggregate = 160
filter_num_skip='auto'
filter_num_aggregate='auto'

stack_num_down=2
stack_num_up=1
activation='ReLU'
output_activation='Sigmoid',
batch_norm=False
pool=True
unpool=True
deep_supervision=False

backbone=None
weights='imagenet'
freeze_backbone=True
freeze_batch_norm=True
name='unet3plus'

weights = "/HDD/etc/unet3p/unet3p_20.h5"
                  
model = unet_3plus_2d(input_size, n_labels, filter_num_down)
model.load_weights(weights)


from keras_unet_collection import losses

def hybrid_loss(y_true, y_pred):
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim

from glob import glob
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 증가 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
from tensorflow import keras
import keras_unet_collection.utils as utils

def input_data_process(input_array):
    '''converting pixel vales to [0, 1]'''
    return input_array/255.

def target_data_process(target, num_classes):
    # '''Converting tri-mask of {1, 2, 3} to three categories.'''
    # return keras.utils.to_categorical(target_array)

    # (bs, h, w, 1) -> (bs, h, w), squeeze를 통해 채널 차원 제거
    target = np.squeeze(target, axis=-1)
    
    # (bs, h, w) -> (bs, h, w, num_classes)으로 one-hot 인코딩
    target_one_hot = keras.utils.to_categorical(target, num_classes=num_classes)
    
    return target_one_hot

input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/patches'
mask_input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/masks'

# sample_names = np.array(sorted(glob(input_dir + '/*.bmp')))
# label_names = np.array(sorted(glob(mask_input_dir + '/*.bmp')))

# L = len(sample_names)
# ind_all = utils.shuffle_ind(L)

# L_train = int(0.8*L)
# L_valid = int(0.1*L)
# L_test = int(0.005*L)
# ind_train = ind_all[:L_train]
# ind_valid = ind_all[L_train:L_train+L_valid]
# ind_test = ind_all[L_train+L_valid:]
# print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))

# test_input = input_data_process(utils.image_to_array(sample_names[ind_test], size=512, channel=3))
# test_target = target_data_process(utils.image_to_array(label_names[ind_test], size=512, channel=1), n_labels)

# N_epoch = 30 # number of epoches
# N_sample = 2 # number of samples per batch

img_file = ['/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/patches/1_122111915384037_8_Virtual_Inner_Short1_22.bmp']
mask_file = ['/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/masks/1_122111915384037_8_Virtual_Inner_Short1_22.bmp']

input_img = input_data_process(utils.image_to_array(img_file, size=512, channel=3))
target_img = target_data_process(utils.image_to_array(mask_file, size=512, channel=1), n_labels)

def compute_loss(y_true, y_pred):
    return hybrid_loss(y_true, y_pred)

pred = model.predict(input_img)
target = target_img

print('Testing set cross-entropy loss = {}'.format(np.mean(keras.losses.categorical_crossentropy(target, pred))))
print('Testing set focal Tversky loss = {}'.format(np.mean(losses.focal_tversky(target, pred))))
print('Testing set IoU loss = {}'.format(np.mean(losses.iou_seg(target, pred))))

import cv2
import imgviz

color_map = imgviz.label_colormap(50)
pred = tf.math.argmax(pred[0], axis=-1).numpy().astype(np.uint8)
pred = color_map[pred].astype(np.uint8)
cv2.imwrite('/HDD/etc/unet3p/pred.png', pred)


# test_loss = []
# for step in range(int(L_test/N_sample)):
#     print(f"\r test: ({step}/{int(L_test/N_sample)}) > {str(test_loss[-1]) if len(test_loss) != 0 else ''}", end="")

#     y_pred = model([test_input[N_sample*step:N_sample*(step + 1)]], training=False)
    
#     # 손실 계산
#     loss = sum([compute_loss(test_target[N_sample*step:N_sample*(step + 1)], y_pred[i]) * loss_weight 
#                 for i, loss_weight in enumerate([0.25, 0.25, 0.25, 0.25, 1.0])])
#     test_loss.append(loss.numpy())

# test_loss = np.mean(test_loss)
# print('val loss: ', test_loss)
