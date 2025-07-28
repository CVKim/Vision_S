import onnxruntime as ort
import numpy as np
from glob import glob 
import cv2
import os.path as osp 
import os
import imgviz
from copy import deepcopy


def infer_onnx(onnx_model_path, input_dir, output_dir):
    session = ort.InferenceSession(onnx_model_path)

    img_files = glob(osp.join(input_dir, '*.bmp'))
    roi = [220, 60, 1340, 828]
    save_raw = True
    color_map = imgviz.label_colormap(50)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    print(f"Input name: {input_name}, shape: {input_shape}, type: {input_type}")

    for img_file in img_files:
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        img = cv2.imread(img_file)
        img = img[roi[1]:roi[3], roi[0]:roi[2]]
        vis_img = deepcopy(img)
        img = img.astype(np.float32)#/255
        img = np.expand_dims(img, 0)
        
        outputs = session.run(None, {input_name: img})
        for i, output in enumerate(outputs):
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


if __name__ == '__main__':
    onnx_model_path = '/DeepLearning/etc/_athena_tests/benchmark/talos/python/deeplabv3plus/export/tenneco_outer.onnx'
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/talos/1st/images_diff_from_python'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/onnx/deeplabv3plus/1st/diff_from_python'
    
    infer_onnx(onnx_model_path, input_dir, output_dir)