from visionsuite.utils.metrics.labelme2metcis import labelme2metrics
from visionsuite.utils.metrics.preds2metrics import preds2metrics
from visionsuite.utils.metrics.metrics import get_performance
from visionsuite.utils.metrics.save import save_pf_by_image_to_excel, save_df_by_class_to_pdf, save_false_images
import os.path as osp


input_dir = '/HDD/etc/curation/mr/clustered_dataset_level7/test'
output_dir = '/HDD/etc/curation/mr/outputs/SEGMENTATION/6_24_16_43_32/test/exp'
# output_dir = '/HDD/etc/curation/tenneco/outputs/SEGMENTATION/infobatch/test/exp'

# input_dir = '/HDD/etc/curation/tenneco/clustered_dataset_level7/test'
# # output_dir = '/HDD/etc/curation/tenneco/outputs/SEGMENTATION/curation_v7/test/exp'
# output_dir = '/HDD/etc/curation/tenneco/outputs/SEGMENTATION/original/test/exp'
# # output_dir = '/HDD/etc/curation/tenneco/outputs/SEGMENTATION/infobatch/test/exp'

ground_truths, class2idx = labelme2metrics(input_dir)
# class2idx = {'DUST': 0, "STABBED": 1}
# preds_json = f'/DeepLearning/etc/_athena_tests/benchmark/mr/plate/bottom/outputs/SEGMENTATION/{model_name}/test/exp_conf0.7/preds.json'
preds_json = osp.join(output_dir, 'preds.json')
detections, class2idx = preds2metrics(preds_json, class2idx, nms=0.2)
class2idx = {'EDGE_STABBED': 0, 'DUST': 1, 'STABBED': 2}
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

pf = get_performance(detections, ground_truths, [-1] + list(classes), iou_threshold, shape_type='polygon')
pf_by_image = pf['by_image']
pf_by_class = pf['by_class']

print('* by image: ', pf['by_image'])
print('* by class: ', pf['by_class'])
txt = open(osp.join(output_dir, 'map.txt'), 'w')
for by_class in pf['by_class'][1:-1]:
    txt.write(f"class: {by_class['class']} > precision: {by_class['precision']}, recall: {by_class['recall']}, ap: {by_class['ap']}\n")

txt.write(f"mAP: {pf['by_class'][-1]['map']}\n")
txt.close()

save_false_images(pf_by_image,  osp.join(output_dir, 'vis'), osp.join(output_dir, 'false_images'))
save_pf_by_image_to_excel(pf_by_image, osp.join(output_dir, 'pf_by_image.xlsx'), idx2class)



# save_df_by_class_to_pdf(pf_by_class, osp.join(output_dir, 'pf_by_class.pdf'), idx2class)

