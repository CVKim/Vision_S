import os 
import os.path as osp 
import glob 
import json 
import cv2
from tqdm import tqdm
from shutil import copyfile
import numpy as np
from visionsuite.utils.helpers import get_filename

from visionsuite.utils.dataset.formats.labelme.utils import get_mask_from_labelme


def labelme2mask(input_dir, output_dir, class2label, input_format, modes, width=None, height=None, vis=False, output_format='bmp', rois=[[]], one_class=False):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    for mode in modes:
        _output_dir = osp.join(output_dir, mode)
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
            
        if vis:
            vis_dir = osp.join(_output_dir, 'vis')
            if not osp.exists(vis_dir):
                os.mkdir(vis_dir)
            
        json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
        
        for json_file in tqdm(json_files, desc=mode):
            img_file = osp.splitext(json_file)[0] + f'.{input_format}'
            img = cv2.imread(img_file)
            filename = get_filename(json_file, False)
            with open(json_file, 'r') as jf:
                anns = json.load(jf)
                
            if width is None and height is None:
                width, height = anns['imageWidth'], anns['imageHeight']
                
            if rois == [[]]:
                rois = [[0, 0, width, height]]
            mask = get_mask_from_labelme(json_file, width, height, class2label, 
                                        format='opencv', one_class=one_class)
            import numpy as np
            mask_output_dir = osp.join(_output_dir, 'masks')
            if not osp.exists(mask_output_dir):
                os.mkdir(mask_output_dir)
                        
            image_output_dir = osp.join(_output_dir, 'images')
            if not osp.exists(image_output_dir):
                os.mkdir(image_output_dir)
                
            for roi in rois:
                roi_img = img[roi[1]:roi[3], roi[0]:roi[2]]
                roi_mask = mask[roi[1]:roi[3], roi[0]:roi[2]]
                
                cv2.imwrite(osp.join(mask_output_dir, filename + f'.{output_format}'), roi_mask)
                cv2.imwrite(osp.join(image_output_dir, filename + f'.{output_format}'), roi_img)
                # copyfile(img_file, osp.join(image_output_dir, filename + f'.{output_format}'))
                
                if vis:
                    import numpy as np
                    import imgviz 
                    
                    vis_img = np.zeros((roi[3] - roi[1], (roi[2] - roi[0])*2, 3))
                    vis_img[:, :roi[2] - roi[0], :] = roi_img
                    color_map = imgviz.label_colormap(50)
                    vis_img[:, roi[2] - roi[0]:, :] = color_map[roi_mask.astype(np.uint8)].astype(np.uint8) 
                    
                    cv2.imwrite(osp.join(vis_dir, filename + '.png'), vis_img)
        
            
if __name__ == '__main__':
    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset'
    output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_mask_dataset_one_class'
    class2label = {'chamfer_mark': 1, 'line': 2, 'mark': 3}
    modes = ['train', 'val']
    # width, height = 1120, 768
    width, height = None, None
    vis = False
    output_format = 'png'
    input_format = 'bmp'
    roi = [[220, 60, 1340, 828]]
    one_class = 255

    labelme2mask(input_dir, output_dir, class2label, input_format, modes, width, height, vis, output_format, roi, one_class)

