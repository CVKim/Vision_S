import os 
import os.path as osp 
from glob import glob 
import json 
from tqdm import tqdm
from visionsuite.utils.dataset.formats.labelme.utils import init_labelme_json, add_labelme_element


def neuro2labelme_seg():
    input_dir = '/DeepLearning/research/data/unittests/unit_cost_test/neurocle/split_mr/results/val_results'
    output_dir = osp.join(input_dir, 'labelme')
    os.makedirs(output_dir, exist_ok=True)
    json_file = osp.join(input_dir, 'val_labeling.json')
    with open(json_file, 'r') as jf:
        anns = json.load(jf)

    for data in tqdm(anns['data']):

        w, h = data['width'], data['height']    
        filename = data['fileName'].split('.')[0]
        region_label = data['regionLabel']
        _labelme = init_labelme_json(filename + '.bmp', w, h)
        
        for label in region_label:
            
            new_points = []
            for point in label['points']:
                new_points.append([point[0], point[1]])
            _labelme = add_labelme_element(_labelme, shape_type='polygon', 
                                        label=label['className'], 
                                        points=new_points)


        with open(os.path.join(output_dir, filename + ".json"), "w") as jsf:
            json.dump(_labelme, jsf)
        
        

def neuro2labelme_det():
    input_dir = '/DeepLearning/research/data/unittests/unit_cost_test/neurocle/split_interojo_dataset/results/test_restults/'
    output_dir = osp.join(input_dir, 'labelme')
    os.makedirs(output_dir, exist_ok=True)
    json_file = osp.join(input_dir, 'obd_det-train_labeling.json')
    with open(json_file, 'r') as jf:
        anns = json.load(jf)

    for data in tqdm(anns['data']):

        w, h = data['width'], data['height']    
        filename = data['fileName'].split('.')[0]
        region_label = data['regionLabel']
        _labelme = init_labelme_json(filename + '.bmp', w, h)
        
        for label in region_label:
            
            new_points = [[label['x'], label['y']], [label['x'] + label['width'], label['x'] + label['height']]]
            _labelme = add_labelme_element(_labelme, shape_type='rectangle', 
                                        label=label['className'], 
                                        points=new_points)


        with open(os.path.join(output_dir, filename + ".json"), "w") as jsf:
            json.dump(_labelme, jsf)
        
def neuro2labelme_det():
    input_dir = '/DeepLearning/research/data/unittests/unit_cost_test/neurocle/split_interojo_dataset/results/val_results'
    output_dir = osp.join(input_dir, 'labelme')
    os.makedirs(output_dir, exist_ok=True)
    json_file = osp.join(input_dir, 'val_labeling.json')
    with open(json_file, 'r') as jf:
        anns = json.load(jf)

    for data in tqdm(anns['data']):

        w, h = data['width'], data['height']    
        filename = data['fileName'].split('.')[0]
        region_label = data['regionLabel']
        _labelme = init_labelme_json(filename + '.bmp', w, h)
        
        for label in region_label:
            
            new_points = [[label['x'], label['y']], [label['x'] + label['width'], label['x'] + label['height']]]
            _labelme = add_labelme_element(_labelme, shape_type='rectangle', 
                                        label=label['className'], 
                                        points=new_points)


        with open(os.path.join(output_dir, filename + ".json"), "w") as jsf:
            json.dump(_labelme, jsf)
        

if __name__ == '__main__':
    # neuro2labelme_seg()
    neuro2labelme_det()

