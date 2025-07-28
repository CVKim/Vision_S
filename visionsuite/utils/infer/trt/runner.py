import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
from modeler import TrtModeler
from copy import deepcopy
import imgviz 


class TrtRunner:
    def __init__(self,
                 trt_file,
                 warm_up=True):

        self.trt_file = trt_file
        self.input_shape = None
        self.input_names = []
        self.output_shape = None
        self.output_names = []
        self._warm_up = warm_up
        self._warm_up_done = False

    def load_trt(self):
        TrtModeler.pycuda_autoinit()
        self.trt_model = TrtModeler(self.trt_file)
        self.input_shape = self.trt_model.engine.get_binding_shape(0)
        self.output_shape = self.trt_model.engine.get_binding_shape(1)

    def __call__(self, x: np.ndarray):

        if self._warm_up and not self._warm_up_done:
            print("> Warming up .......")
            for idx in range(3):
                im = np.zeros(self.input_shape)
                self.trt_model(im, self.input_shape[0])
                
            print("> Warmed up !!!!!!!")
            self._warm_up_done = True 
            
        return self.trt_model(x, self.input_shape[0])

if __name__ == '__main__':
    
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/talos/1st/images_diff_from_python'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/talos/python/deeplabv3plus/test_trt/deeplabv3plus/1st/diff_from_python'
    roi = [220, 60, 1340, 828]
    save_raw = True
    color_map = imgviz.label_colormap(50)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
    trt_path = '/DeepLearning/etc/_athena_tests/benchmark/talos/python/deeplabv3plus/export/tenneco_outer.trt'
    trt_runner = TrtRunner(trt_path)
    trt_runner.load_trt()
    
    img_files = glob(osp.join(input_dir, '*.bmp'))
    for img_file in img_files:
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        img = cv2.imread(img_file)
        img = img[roi[1]:roi[3], roi[0]:roi[2]]
        vis_img = deepcopy(img)
        img = img.astype(np.float32)#/255
        img = np.expand_dims(img, 0)
        
        outputs = trt_runner(img)
        for i, output in enumerate(outputs):
            output = output.reshape(trt_runner.output_shape)
            if save_raw:
                raw_dir = osp.join(output_dir, 'raw')
                if not osp.exists(raw_dir):
                    os.mkdir(raw_dir)
                np.save(osp.join(raw_dir, filename + f'.npy'), output[0])
                
            vis_dir = osp.join(output_dir, 'vis')
            if not osp.exists(vis_dir):
                os.mkdir(vis_dir)
                
            vis_output = np.argmax(output[0], axis=2).astype(np.uint8)
            vis_output = color_map[vis_output].astype(np.uint8)
            vis_img = cv2.addWeighted(vis_img, 0.4, vis_output, 0.6, 0)
            cv2.imwrite(osp.join(vis_dir, filename + '.png'), vis_img)
        