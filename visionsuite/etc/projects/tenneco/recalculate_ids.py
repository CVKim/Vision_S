from glob import glob 
import os.path as osp 
import os 
import json

iou = 0.05
area = 20

# input_dir = '/HDD/etc/repeatablility/deeplabv3plus_2/filenames'
# output_dir = '/HDD/etc/repeatablility/deeplabv3plus_2'
# total = 11214/14
input_dir = '/HDD/etc/repeatablility/2nd/deeplabv3plus_2/filenames'
output_dir = '/HDD/etc/repeatablility/2nd/deeplabv3plus_2'
total = 7714/14

diff_points_txt = osp.join(input_dir, f'IoU-{iou}_Area-{area}_Conf-0_diff_points.txt')
no_diff_no_points_txt = osp.join(input_dir, f'IoU-{iou}_Area-{area}_Conf-0_no_diff_no_points.txt')
no_diff_points_txt = osp.join(input_dir, f'IoU-{iou}_Area-{area}_Conf-0_no_diff_points.txt')

with open(diff_points_txt, "r", encoding="utf-8") as file:
    diff_points_ids = set([line.strip().split("_")[0] for line in file])
    
with open(no_diff_no_points_txt, "r", encoding="utf-8") as file:
    no_diff_no_points_ids = set([line.strip().split("_")[0] for line in file])
    
with open(no_diff_points_txt, "r", encoding="utf-8") as file:
    no_diff_points_ids = set([line.strip().split("_")[0] for line in file])
    
results = {"total_ids": total,
        #    "diff_points_ids": len(diff_points_ids),
        #    "no_diff_no_points_ids": len(no_diff_no_points_ids),
        #    "no_diff_points_ids": len(no_diff_points_ids),
           'diff_ids': len(diff_points_ids),
           'diff_ids (%)': len(diff_points_ids)/total*100
        }
        
new_no_diff_points_ids = set()
for no_diff_points_id in no_diff_points_ids:
    if no_diff_points_id not in diff_points_ids and no_diff_points_id not in new_no_diff_points_ids:
        new_no_diff_points_ids.add(no_diff_points_id)
        
new_no_diff_no_points_ids = set()
for no_diff_no_points_id in no_diff_no_points_ids:
    if no_diff_no_points_id not in diff_points_ids and no_diff_no_points_id not in new_no_diff_no_points_ids and no_diff_no_points_id not in new_no_diff_points_ids:
        new_no_diff_no_points_ids.add(no_diff_no_points_id)

results["no_diff_points_ids"] = len(new_no_diff_points_ids)
results["no_diff_points_ids (%)"] = len(new_no_diff_points_ids)/total*100
results["no_diff_no_points_ids"] = len(new_no_diff_no_points_ids)
results["no_diff_no_points_ids (%)"] = len(new_no_diff_no_points_ids)/total*100
assert results["no_diff_points_ids"] + results["no_diff_no_points_ids"] + results["diff_ids"] == total

with open(osp.join(output_dir, f'results-IoU-{iou}_Area-{area}_Conf-0.txt'), 'w') as file:
    json.dump(results, file)