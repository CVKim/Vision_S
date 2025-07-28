import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from label_smooth import LabelSmoothSoftmaxCEV1
from functools import partial
from operator import itemgetter

# Helper functions
def kl_div(a, b):
    return tf.nn.softmax(b, axis=1) * (tf.nn.log_softmax(b, axis=1) - tf.nn.log_softmax(a, axis=1))

def one_hot2dist(seg):
    seg = tf.convert_to_tensor(seg)
    res = tf.zeros_like(seg, dtype=tf.float32)
    
    for i in range(tf.shape(seg)[0]):
        posmask = tf.cast(seg[i], tf.bool)
        if tf.reduce_any(posmask):
            negmask = tf.logical_not(posmask)
            res_i = tf.where(negmask, tf.cast(distance(negmask.numpy()), tf.float32), 0.0)
            res_i = res_i - tf.where(posmask, tf.cast(distance(posmask.numpy()) - 1, tf.float32), 0.0)
            res = tf.tensor_scatter_nd_update(res, [[i]], [res_i])
    
    return res


def class2one_hot(seg, C):
    seg = tf.expand_dims(seg, axis=0) if len(seg.shape) == 2 else seg
    res = tf.stack([tf.cast(seg == c, tf.int32) for c in range(C)], axis=1)
    return res

class ABL(tf.keras.layers.Layer):
    def __init__(self, isdetach=True, max_N_ratio=1/100, ignore_label=255, label_smoothing=0.2, weight=None, max_clip_dist=20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach = isdetach
        self.max_N_ratio = max_N_ratio
        self.weight_func = lambda w, max_distance=max_clip_dist: tf.clip_by_value(w, 0, max_distance) / max_distance

        self.dist_map_transform = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int64)),
            tf.keras.layers.Lambda(partial(class2one_hot, C=1)),
            tf.keras.layers.Lambda(lambda x: x[0]),  # Replace itemgetter(0) with this
            tf.keras.layers.Lambda(lambda x: tf.py_function(one_hot2dist, [x], tf.float32))
        ])

        if label_smoothing == 0:
            self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
        else:
            self.criterion = LabelSmoothSoftmaxCEV1(
                reduction='none', ignore_index=ignore_label, lb_smooth=label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, h, w, _ = logit.shape
        max_N = tf.cast(h * w * self.max_N_ratio, tf.float32)
        kl_ud = tf.reduce_sum(kl_div(logit[:, 1:, :, :], logit[:, :-1, :, :]), axis=-1, keepdims=True)
        kl_lr = tf.reduce_sum(kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]), axis=-1, keepdims=True)
        kl_ud = tf.pad(kl_ud, [[0, 0], [0, 1], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
        kl_lr = tf.pad(kl_lr, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT', constant_values=0)
        kl_combine = kl_lr + kl_ud

        while True:
            kl_combine_bin = tf.cast(kl_combine > eps, tf.float32)
            if tf.reduce_sum(kl_combine_bin) > max_N:
                eps *= 1.2
            else:
                break

        dilate_weight = tf.ones((3, 3, 1, 1))
        edge2 = tf.nn.conv2d(kl_combine_bin, dilate_weight, strides=[1, 1, 1, 1], padding='SAME')
        edge2 = tf.squeeze(edge2, axis=-1)
        kl_combine_bin = edge2 > 0
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):
        gt_ud = gt[:, 1:, :] - gt[:, :-1, :]
        gt_lr = gt[:, :, 1:] - gt[:, :, :-1]
        gt_ud = tf.pad(gt_ud, [[0, 0], [0, 1], [0, 0]]) != 0
        gt_lr = tf.pad(gt_lr, [[0, 0], [0, 0], [0, 1]]) != 0
        gt_combine = gt_lr | gt_ud
        gt_combine = gt_combine | tf.equal(gt, ignore_label)
        return gt_combine

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        eps = 1e-5
        bound = tf.where(pred_bound)
        n, x, y = tf.unstack(bound, axis=1)
        max_dis = 1e5

        logits = tf.transpose(logits, [0, 2, 3, 1])
        pred_dist_map_d = tf.pad(pred_dist_map, [[0, 0], [1, 1], [1, 1]], mode='CONSTANT', constant_values=max_dis)
        logits_d = tf.pad(logits, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')

        x_range = [1, -1, 0, 0, -1, 1, -1, 1, 0]
        y_range = [0, 0, -1, 1, 1, 1, -1, -1, 0]
        dist_maps = []
        kl_maps = []

        kl_center = tf.gather_nd(logits, tf.stack([n, x, y], axis=1))

        for dx, dy in zip(x_range, y_range):
            dist_now = tf.gather_nd(pred_dist_map_d, tf.stack([n, x+dx+1, y+dy+1], axis=1))
            dist_maps.append(dist_now)

            if dx != 0 or dy != 0:
                logits_now = tf.gather_nd(logits_d, tf.stack([n, x+dx+1, y+dy+1], axis=1))
                if self.isdetach:
                    logits_now = tf.stop_gradient(logits_now)
                kl_map_now = kl_div(kl_center, logits_now)
                kl_map_now = tf.reduce_sum(kl_map_now, axis=1)
                kl_maps.append(kl_map_now)

        dist_maps = tf.stack(dist_maps, axis=0)
        kl_maps = tf.stack(kl_maps[:-1], axis=0)  # Exclude the last element (center)

        direction_gt = tf.argmin(dist_maps, axis=0)
        weight_ce = tf.gather_nd(pred_dist_map, tf.stack([n, x, y], axis=1))

        direction_gt_idx = direction_gt != 8
        direction_gt = tf.boolean_mask(direction_gt, direction_gt_idx)
        kl_maps = tf.transpose(kl_maps, [1, 0])
        direction_pred = tf.boolean_mask(kl_maps, direction_gt_idx)
        weight_ce = tf.boolean_mask(weight_ce, direction_gt_idx)

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = tf.identity(target)
        dist_maps = tf.concat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])], axis=0)
        out = -dist_maps
        out = tf.where(out > 0, out, tf.zeros_like(out))
        return out

    def call(self, logits, target):
        eps = 1e-10
        ph, pw = tf.shape(logits)[2], tf.shape(logits)[3]
        h, w = tf.shape(target)[1], tf.shape(target)[2]

        if ph != h or pw != w:
            logits = tf.image.resize(logits, [h, w], method='bilinear')

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)
        dist_maps = self.get_dist_maps(gt_boundary)
        pred_boundary = self.logits2boundary(logits)

        if tf.reduce_sum(tf.cast(pred_boundary, tf.float32)) < 1:
            return None

        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits)
        loss = self.criterion(direction_pred, direction_gt)
        weight_ce = self.weight_func(weight_ce)
        loss = tf.reduce_mean(loss * weight_ce)

        return loss
    
if __name__ == '__main__':

    import tensorflow as tf
    import numpy as np
    import random
    import os

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


    # Define test dimensions
    n, c, h, w = 1, 2, 100, 100

    # Create ground truth tensor
    gt = np.zeros((n, h, w))
    gt = np.zeros_like(gt)  # Initialize with zeros if not already done

    # Perform the scatter update
    gt[0, 5, 0] = 1.0
    gt[0, 50, 0] = 1.0

    logits = np.random.normal(loc=0.0, scale=4.0, size=(n, c, h, w)).astype(np.float32)
    tf_logits = tf.constant(logits)
    tf_gt = tf.constant(gt)
    # tf.config.set_visible_devices([], 'GPU')
    tf_abl = ABL()
    tf_loss = tf_abl(tf_logits, tf_gt)

    print("TensorFlow ABL Loss:", tf_loss.numpy())
    
    from visionsuite.utils.losses.torch.active_boundary_loss import ABL as TorchABL
    import torch 
    
    torch_logits = torch.from_numpy(logits)
    torch_gt = torch.from_numpy(gt).to(torch.int64)
    torch_abl = TorchABL()
    torch_loss = torch_abl(torch_logits, torch_gt)
    print("Torch ABL Loss:", tf_loss.numpy())
    
    
    
