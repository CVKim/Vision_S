from visionsuite.utils.metrics.labelme2metcis import labelme2metrics
from visionsuite.utils.metrics.preds2metrics import preds2metrics
from visionsuite.utils.metrics.metrics import get_performance
from visionsuite.utils.metrics.save import save_pf_by_image_to_excel, save_df_by_class_to_pdf
import os.path as osp

# model_name = 'tf_deeplabv3plus_epochs100'
# model_name = 'tf_deeplabv3plus_epochs200'
# model_name = 'tf_deeplabv3plus_frozen_epochs100'
# model_name = 'm2f_epochs100'
# model_name = 'cosnet_epochs100'
# model_name = 'pidnet_epochs100'
# model_name = 'gcnet_epochs100'
model_name = 'sam2unet_epochs100'
output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{model_name}/test/exp'
input_dir = '/DeepLearning//etc/_athena_tests/benchmark/tenneco/outer/val'

# # model_name = 'deeplabv3plus'
# # model_name = 'pidnet_epochs300'
# # model_name = 'dinov2_epochs50'
# # model_name = 'm2f_epochs20'
# model_name = 'gcnet_epochs200'

# # output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/mr/plate/bottom/outputs/SEGMENTATION/{model_name}/test/exp'


ground_truths, class2idx = labelme2metrics(input_dir)
# class2idx = {'DUST': 0, "STABBED": 1}
preds_json = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{model_name}/test/exp/preds.json'
detections, class2idx = preds2metrics(preds_json, class2idx, nms=0)
# detections, class2idx = preds2metrics(preds_json, class2idx, nms=0.2)
print(class2idx)

# _detections, _ground_truths = [], []
# for detection in detections:
#     if detection[0] == '1182_1_angle_20241212161927203':
#         _detections.append(detection)
# for ground_truth in ground_truths:
#     if ground_truth[0] == '1182_1_angle_20241212161927203':
#         _ground_truths.append(ground_truth)
# detections = _detections
# ground_truths = _ground_truths

iou_threshold = 0.1
classes = class2idx.values()
idx2class = {idx: _class for _class, idx in class2idx.items()}
print(idx2class)

pf = get_performance(detections, ground_truths, classes, iou_threshold, shape_type='polygon')
pf_by_image = pf['by_image']
pf_by_class = pf['by_class']

print('* by image: ', pf['by_image'])
print('* by class: ', pf['by_class'])

save_pf_by_image_to_excel(pf_by_image, osp.join(output_dir, 'pf_by_image.xlsx'), idx2class)
save_df_by_class_to_pdf(pf_by_class, osp.join(output_dir, 'pf_by_class.pdf'), idx2class)

