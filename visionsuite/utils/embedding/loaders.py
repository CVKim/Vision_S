from glob import glob 
import os.path as osp
import cv2

class ImageIterator:
    def __init__(self, input_dir, transform, 
                 rois=[[]], patch_info={}, image_formats=['bmp']):
        
        self.transform = transform
        self.rois = [[]] if rois is None else rois
        self.patch_info = {} if patch_info is None else patch_info
        
        for image_format in image_formats:
            self.img_files = glob(osp.join(input_dir, f'*.{image_format}'))
        
        self.img = None
        self.filename = None 
        self.roi_idx = 0
        self.patch_idx = 0
        
    def __len__(self):
        return len(self.img_files)    
        
    def __iter__(self):
        
        return self 
    
    def __next__(self):
        
        if self.img is None:
            
            if len(self.img_files) == 0:
                raise StopIteration
            
            img_file = self.img_files.pop()
            self.filename = osp.split(osp.splitext(img_file)[0])[-1]
            self.img = cv2.imread(img_file)
        
        if self.rois[self.roi_idx] != []:
            img = self.img[self.rois[self.roi_idx][1]:self.rois[self.roi_idx][3], 
                           self.rois[self.roi_idx][0]:self.rois[self.roi_idx][2]]
            self.roi_idx += 1
        else:
            img = self.img
            
            
        if self.patch_info != {} and self.patch_info['use']:
            
            dx = int((1.0 - self.patch_info['overlap_ratio']) * self.patch_info['width'])
            dy = int((1.0 - self.patch_info['overlap_ratio']) * self.patch_info['height'])
            
            tl_x, tl_y, br_x, br_y = 0, 0, img.shape[1], img.shape[0]
            _patch_idx = -1
            _done = False
            for y0 in range(tl_y, br_y, dy):
                for x0 in range(tl_x, br_x, dx):
                    
                    _patch_idx += 1
                    if _patch_idx != self.patch_idx:
                        continue
                    
                    if y0 + self.patch_info['height'] > br_y:
                        if self.patch_info['skip_highly_overlapped_tiles']:
                            if (y0 + self.patch_info['height'] - br_y) > (0.6 * self.patch_info['height']):
                                continue
                            else:
                                y = br_y - self.patch_info['height']
                        else:
                            y = br_y - self.patch_info['height']
                    else:
                        y = y0

                    if x0 + self.patch_info['width'] > br_x:
                        if self.patch_info['skip_highly_overlapped_tiles']:
                            if (x0 + self.patch_info['width'] - br_x) > (0.6 * self.patch_info['width']):
                                continue
                            else:
                                x = br_x - self.patch_info['width']
                        else:
                            x = br_x - self.patch_info['width']
                    else:
                        x = x0

                    xmin, xmax, ymin, ymax = (
                        x,
                        x + self.patch_info['width'],
                        y,
                        y + self.patch_info['height'],
                    )  # patch coordinates
                    
                    img = img[ymin:ymax, xmin:xmax]
                    self.patch_idx += 1
                    _done = True
                    break
                if _done:
                    break
            
        if self.roi_idx == len(self.rois) or self.rois == [[]]:
            self.img = None 

        return img, self.filename
        
        
if __name__ == '__main__':
    import os 
    input_dir = '/HDD/datasets/projects/mr/split_dataset_unit/train'
    transform = None
    rois = [[]]
    # patch_info = {'use': True, 'width': 512, 'height': 512, 
    #               'overlap_ratio': 0.2, 'skip_highly_overlapped_tiles': False}
    patch_info = {}
    iterator = ImageIterator(input_dir=input_dir, transform=transform, rois=rois, patch_info=patch_info)
    
    output_dir = '/HDD/etc/test/outputs'
    os.makedirs(output_dir, exist_ok=True)
    for (img, filename) in iterator:
        
        cv2.imwrite(osp.join(output_dir, filename + '.png'), img)
        