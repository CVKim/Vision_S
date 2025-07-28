import os.path as osp
import glob 
import json
from visionsuite.utils.metrics.geometry import merge_polygons



def preds2metrics(preds_json, class2idx, nms=0):
    with open(preds_json, 'r') as jf:
        anns = json.load(jf)
    
    detections = []
    for filename, ann in anns.items():

        if ann['idx2xyxys'] == {}:
            detections.append([filename, -1, -1, ()])
            
        for _class, val in ann['idx2xyxys'].items():
            if 'bbox' in val:
                for box, conf in zip(val['bbox'], val['confidence']):
                    if ann['idx2class'][_class] not in class2idx:
                        class2idx[ann['idx2class'][_class]] = len(class2idx)
                    detections.append([filename, int(class2idx[ann['idx2class'][_class]]), float(conf), (box[0][0], box[0][1], box[1][0], box[1][1])])
            elif 'polygon' in val:
                if 'confidence' in val:
                    for box, conf in zip(val['polygon'], val['confidence']):
                        if ann['idx2class'][_class] not in class2idx:
                            class2idx[ann['idx2class'][_class]] = len(class2idx)
                        detections.append([filename, int(class2idx[ann['idx2class'][_class]]), float(conf), tuple([_point for __point in box for _point in __point])])
                else:
                    if nms:
                        nms_polygons = merge_polygons(val['polygon'], nms)
                        for polygon in nms_polygons:
                            if ann['idx2class'][_class] not in class2idx:
                                class2idx[ann['idx2class'][_class]] = len(class2idx)
                                
                            detections.append([filename, int(class2idx[ann['idx2class'][_class]]), None, 
                                               tuple(coord for points in polygon for coord in points)])
                    
                    else:    
                        for polygons in zip(val['polygon']):
                            if ann['idx2class'][_class] not in class2idx:
                                class2idx[ann['idx2class'][_class]] = len(class2idx)
                                
                            detections.append([filename, int(class2idx[ann['idx2class'][_class]]), None, 
                                               tuple(coord for polygon in polygons for points in polygon for coord in points)])

    return detections, class2idx
        
    
if __name__ == '__main__':
    preds_json = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/tf_deeplabv3plus_100epochs/test/preds.json'
    class2idx = {"CHAMFER_MARK": 1, "LINE": 2, "MARK": 3}
    detections = preds2metrics(preds_json, class2idx)
    print(detections)