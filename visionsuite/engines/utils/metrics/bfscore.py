import numpy as np
from skimage import segmentation
from scipy.ndimage import distance_transform_edt
import os.path as osp
import glob 
import os
import cv2



def ex_find_boundries(labels):
    boundaries = segmentation.find_boundaries(labels, mode='thick')
    print("thick: \n", boundaries.astype(np.uint8))
    '''
        [[0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 1 1 0 0]
        [0 0 1 1 1 1 1 1 1 0]
        [0 1 1 1 1 1 0 1 1 0]
        [0 1 1 0 1 1 0 1 1 0]
        [0 1 1 1 1 1 0 1 1 0]
        [0 0 1 1 1 1 1 1 1 0]
        [0 0 0 0 0 1 1 1 0 0]
        [0 0 0 0 0 0 0 0 0 0]]
    '''
    
    boundaries = segmentation.find_boundaries(labels, mode='inner')
    print("inner: \n", boundaries.astype(np.uint8))
    '''
        [[0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 1 1 0 0]
        [0 0 1 1 1 1 0 1 0 0]
        [0 0 1 0 1 1 0 1 0 0]
        [0 0 1 1 1 1 0 1 0 0]
        [0 0 0 0 0 1 1 1 0 0]
        [0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0]]
    '''

    boundaries = segmentation.find_boundaries(labels, mode='outer')
    print("outer: \n", boundaries.astype(np.uint8))
    '''
        [[0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 1 1 0 0]
        [0 0 1 1 1 1 0 0 1 0]
        [0 1 0 0 1 1 0 0 1 0]
        [0 1 0 0 1 1 0 0 1 0]
        [0 1 0 0 1 1 0 0 1 0]
        [0 0 1 1 1 1 0 0 1 0]
        [0 0 0 0 0 1 1 1 0 0]
        [0 0 0 0 0 0 0 0 0 0]]
    '''
    
def ex_distance_transform_edt(labels, tolerance):
    
    boundaries = segmentation.find_boundaries(labels, mode='thick')
    print("~boundaries: \n", (~boundaries).astype(np.uint8))
    '''
        [[1 1 1 1 1 1 1 1 1 1]
        [1 1 1 1 1 0 0 0 1 1]
        [1 1 0 0 0 0 0 0 0 1]
        [1 0 0 0 0 0 1 0 0 1]
        [1 0 0 1 0 0 1 0 0 1]
        [1 0 0 0 0 0 1 0 0 1]
        [1 1 0 0 0 0 0 0 0 1]
        [1 1 1 1 1 0 0 0 1 1]
        [1 1 1 1 1 1 1 1 1 1]]
    '''
    edt = distance_transform_edt(~boundaries)
    print("\nedt: \n", edt.astype(np.uint8))
    '''
        [[2 2 2 2 1 1 1 1 1 2]
        [2 1 1 1 1 0 0 0 1 1]
        [1 1 0 0 0 0 0 0 0 1]
        [1 0 0 0 0 0 1 0 0 1]
        [1 0 0 1 0 0 1 0 0 1]
        [1 0 0 0 0 0 1 0 0 1]
        [1 1 0 0 0 0 0 0 0 1]
        [2 1 1 1 1 0 0 0 1 1]
        [2 2 2 2 1 1 1 1 1 2]]
    '''
    
    edt = edt <= tolerance
    print("\nedt <= tolerance: \n", edt.astype(np.uint8))
    '''
        [[0 0 0 0 0 1 1 1 0 0]
        [0 0 1 1 1 1 1 1 1 0]
        [0 1 1 1 1 1 1 1 1 1]
        [1 1 1 1 1 1 1 1 1 1]
        [1 1 1 1 1 1 1 1 1 1]
        [1 1 1 1 1 1 1 1 1 1]
        [0 1 1 1 1 1 1 1 1 1]
        [0 0 1 1 1 1 1 1 1 0]
        [0 0 0 0 0 1 1 1 0 0]]
    '''
  
def bfscore(pred, gt, tolerances=None, mode="inner"):
    """
    - True Positive (TP): 예측 경계와 실제 경계가 허용 오차 내에 있을 때.
    - False Positive (FP): 예측 경계가 실제 경계 근처에 없을 때.
    - False Negative (FN): 실제 경계가 예측 경계 근처에 없을 때.
    """

    if isinstance(tolerances, int):
        tolerances = {"pred": tolerances, "gt": tolerances}
    elif isinstance(tolerances, dict):
        assert "pred" in tolerances, ValueError(f"Tolerance must habe pred")
        assert "gt" in tolerances, ValueError(f"Tolerance must habe gt")
    else:
        tolerances = {"pred": 1, "gt": 1}

    pred_boundaries = segmentation.find_boundaries(pred, mode=mode)
    gt_boundaries = segmentation.find_boundaries(gt, mode=mode)

    pred_edt = distance_transform_edt(~pred_boundaries)
    gt_edt = distance_transform_edt(~gt_boundaries)

    pred_edt = pred_edt <= tolerances["pred"]
    gt_edt = gt_edt <= tolerances["gt"]

    tp = np.sum(pred_boundaries & gt_edt)
    fp = np.sum(pred_boundaries & ~gt_edt)
    fn = np.sum(gt_boundaries & ~pred_edt)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    return {"f1_score": f1_score, "precision": precision, "recall": recall}


def bfscores_by_channel(pred, gt, tolerances=None, mode="inner"):
    assert pred.ndim == 3, ValueError(f"Pred must be 3D, not {pred.shape}")
    assert gt.ndim == 3, ValueError(f"Gt must be 3D, not {gt.shape}")
    assert pred.shape[-1] == gt.shape[-1], ValueError(
        f"Pred({pred.shape[-1]}) and Gt({gt.shape[-1]}) must have same number of channels"
    )

    scores = {}
    for ch_idx in range(1, pred.shape[-1]):
        score = bfscore(
            pred[:, :, ch_idx], gt[:, :, ch_idx], tolerances=tolerances, mode=mode
        )
        scores[ch_idx] = {
            "f1_score": score["f1_score"],
            "precision": score["precision"],
            "recall": score["recall"],
        }

    return scores


def bfscores_by_batch(pred, gt, tolerances=None, mode="inner"):

    pred = pred.numpy()
    gt = gt.numpy()

    assert pred.ndim == 4, ValueError(f"Pred must be 4D, not {pred.shape}")
    assert gt.ndim == 4, ValueError(f"Gt must be 4D, not {gt.shape}")
    assert pred.shape[-1] == gt.shape[-1], ValueError(
        f"Pred({pred.shape[-1]}) and Gt({gt.shape[-1]}) must have same number of channels"
    )

    total_scores = []
    for batch_idx in range(0, pred.shape[0]):
        scores = bfscores_by_channel(
            pred[batch_idx], gt[batch_idx], tolerances=tolerances, mode=mode
        )
        total_scores.append(scores)

    scores = {}
    for total_score in total_scores:
        for label, score in total_score.items():
            scores[label] = {
                "f1_score": tf.convert_to_tensor(np.mean(score["f1_score"])),
                "precision": tf.convert_to_tensor(np.mean(score["precision"])),
                "recall": tf.convert_to_tensor(np.mean(score["recall"])),
            }

    return scores


    
if __name__ == '__main__':
    # ## For example ---------------------------------------------------------- 
    # labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    #                     [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    #                     [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    #                     [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    #                     [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    # ex_find_boundries(labels)
    # ex_distance_transform_edt(labels, tolerance=0)
    # ## ----------------------------------------------------------------------

    img_file = '/HDD/_projects/etc/bfscore_python-master/data/pred_0.png'
    mask_file = '/HDD/_projects/etc/bfscore_python-master/data/gt_0.png'
    assert osp.exists(img_file), ValueError(f"There is no such image: {img_file}")
    assert osp.exists(mask_file), ValueError(f"There is no such image: {mask_file}")

    tolerance = 2

    pred = cv2.imread(img_file)
    gt = cv2.imread(mask_file)

    scores = bfscores_by_channel(pred, gt)
    print(scores)