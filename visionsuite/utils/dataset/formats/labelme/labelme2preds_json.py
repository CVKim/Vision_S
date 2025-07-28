import os 
import os.path as osp 
import glob 
import json 
from tqdm import tqdm
from visionsuite.utils.helpers import get_filename


def labelme2preds_json(input_dir, output_dir, class2idx):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(osp.join(input_dir, '*.json'))
    
    results = {}
    for json_file in tqdm(json_files):
        filename = get_filename(json_file, False)
        img_file = osp.splitext(json_file)[0] + '.bmp'
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        idx2xyxys = {}
        for ann in anns:
            if ann['label'] == 'background':
                continue
            _class = ann['label'].upper()
            if class2idx[_class] not in idx2xyxys:
                idx2xyxys[class2idx[_class]] = {'polygon': []}
                
            idx2xyxys[class2idx[_class]]['polygon'].append(ann['points'])
            
        results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
                
    with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    # model_name = 'sam2_epochs200'
    # input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/mr/plate/bottom/outputs/SEGMENTATION/{model_name}/test/exp/labels'
    # output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/mr/plate/bottom/outputs/SEGMENTATION/{model_name}/test/exp'
    # classes = ['STABBED', 'DUST']
    # # input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{model_name}/test/exp/labels'
    # # output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{model_name}/test/exp'
    # # classes = ['CHAMFER_MARK', 'LINE', 'MARK']
    # idx2class = {idx: cls for idx, cls in enumerate(classes)}
    # class2idx = {cls: idx for idx, cls in enumerate(classes)}

    # labelme2preds_json(input_dir, output_dir, class2idx)

    input_dir = f'/HDD/etc/curation/mr/outputs/SEGMENTATION/6_24_16_43_32/test/exp/labels'
    output_dir = f'/HDD/etc/curation/mr/outputs/SEGMENTATION/6_24_16_43_32/test/exp'
    classes = ['STABBED', 'DUST', 'EDGE_STABBED']
    # input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{model_name}/test/exp/labels'
    # output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/{model_name}/test/exp'
    # classes = ['CHAMFER_MARK', 'LINE', 'MARK']
    idx2class = {idx: cls for idx, cls in enumerate(classes)}
    class2idx = {cls: idx for idx, cls in enumerate(classes)}

    labelme2preds_json(input_dir, output_dir, class2idx)

