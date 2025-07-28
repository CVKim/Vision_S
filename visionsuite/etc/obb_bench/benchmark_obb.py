from visionsuite.utils.metrics.labelme2metcis import labelme2metrics
from visionsuite.utils.metrics.preds2metrics import preds2metrics
from visionsuite.utils.metrics.metrics import get_performance
from visionsuite.utils.metrics.save import save_pf_by_image_to_excel, save_df_by_class_to_pdf
import os.path as osp

# ######################## STD #####################################################################
# model_name = 'std'
# backbone = 'hivit'

# ######################## RTMDet #####################################################################
# model_name = 'rtmdet'
# backbone = 'large'

######################## yolov8-obb #####################################################################
model_name = 'yolov8'
backbone = 'l_dfl_0.2_norad'

# ######################## yolov10-obb #####################################################################
# model_name = 'yolov10'
# backbone = 'l'

output_dir = f'/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/tests/sfaw/{model_name}_{backbone}'

input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/sfaw'
# input_dir = '/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/dataset/split_dataset_doosan/val'
ground_truths, class2idx = labelme2metrics(input_dir)
print(class2idx)

preds_json = f'/HDD/_projects/benchmark/obb_detection/doosan_cj_rich/tests/sfaw/{model_name}_{backbone}/preds.json'
detections, class2idx = preds2metrics(preds_json, class2idx)
print(class2idx)

# _detections, _ground_truths = [], []
# for detection in detections:
#     if detection[0] == '14_124060517095294_1_rgb':
#         _detections.append(detection)
# for ground_truth in ground_truths:
#     if ground_truth[0] == '14_124060517095294_1_rgb':
#         _ground_truths.append(ground_truth)
# detections = _detections
# ground_truths = _ground_truths

iou_threshold = 0.5
classes = class2idx.values()
idx2class = {idx: _class for _class, idx in class2idx.items()}

pf = get_performance(detections, ground_truths, classes, iou_threshold, shape_type='polygon')
pf_by_image = pf['by_image']
pf_by_class = pf['by_class']

print('* by image: ', pf['by_image'])
print('* by class: ', pf['by_class'])

save_pf_by_image_to_excel(pf_by_image, osp.join(output_dir, 'pf_by_image.xlsx'), idx2class)
save_df_by_class_to_pdf(pf_by_class, osp.join(output_dir, 'pf_by_class.pdf'), idx2class)

