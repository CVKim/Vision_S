import os 
import os.path as osp 
from glob import glob 
import json 
from tqdm import tqdm
from visionsuite.utils.dataset.formats.labelme.utils import init_labelme_json, add_labelme_element

orders = ['1', '2', '3']
# case = '1st'
case = '2nd'
defects = ['오염', '경계성', '딥러닝', 'repeated_ng', 'repeated_ok']
for defect in defects:
    input_dir = f'/HDD/etc/repeatablility/talos2/{case}/benchmark/neurocle/{defect}'
    output_dir = f'/HDD/etc/repeatablility/talos2/{case}/benchmark/neurocle/{defect}'

    os.makedirs(output_dir, exist_ok=True)


    for order in orders:

        json_file = osp.join(input_dir, f'{order}.json')
        with open(json_file, 'r') as jf:
            anns = json.load(jf)
        w, h = 1120, 768
            
        for data in tqdm(anns['data'], desc=f'{str(order)}: '):
            
            filename = data['fileName'].split('.')[0]
            region_label = data['regionLabel']
            _labelme = init_labelme_json(filename + '.bmp', w, h)
            
            for label in region_label:
                
                new_points = []
                for point in label['points']:
                    new_points.append([point[0] + 220, point[1]+ 60])
                _labelme = add_labelme_element(_labelme, shape_type='polygon', 
                                            label=label['className'], 
                                            points=new_points)


            if order == '1':
                _output_dir = osp.join(output_dir, f'exp/labels')
            else:     
                _output_dir = osp.join(output_dir, f'exp{order}/labels')

            os.makedirs(_output_dir, exist_ok=True)

            with open(os.path.join(_output_dir, filename + ".json"), "w") as jsf:
                json.dump(_labelme, jsf)
            
            
                
