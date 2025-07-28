#!/usr/bin/env python3
import os.path as osp
import os
from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register_module()
class MaskDataset(BaseDataset):
    def __init__(
        self,
        classes,
        img_suffix,
        seg_map_suffix,
        **kwargs,
    ):
        self.CLASSES = classes
        super(MaskDataset, self).__init__(**kwargs)
        
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.img_infos = self.load_annotations(
            img_dir=self.img_dir,
            ann_dir=self.ann_dir,
        )

    def load_annotations(self, img_dir, ann_dir ):
        img_infos = []
        for img_name in os.listdir(img_dir):
            if img_name.endswith(self.img_suffix):
                img_info = dict(filename=img_name)
                img_info['ann'] = dict(seg_map=osp.join(ann_dir, img_name))
                img_infos.append(img_info)
        
        return img_infos
        
    def pre_pipeline(self, results):
        results["seg_fields"] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir