# import argparse
# import yaml
# import numpy as np

# from athena.src.data import set_roi_params
# from athena.src.data import set_patch_params
# from athena.src.tasks.segmentation.utils.params.db import set_db_params

# from athena.src.data import get_images_info

# class PatchInfo:
#     def __init__(self):
#         self.cfgs = argparse.Namespace()
#         self._vars = argparse.Namespace()

#     def alg_set_cfgs(self, recipe):

#         _cfgs = vars(self.cfgs)
#         if recipe is not None:
#             if isinstance(recipe, str):
#                 with open(recipe) as f:
#                     recipe = yaml.safe_load(f)

#             for key, val in recipe.items():
#                 _cfgs[key] = val

#         set_roi_params(self.cfgs, self._vars)
#         set_patch_params(self.cfgs, self._vars)
#         set_db_params(self.cfgs, self._vars)

#         if hasattr(self.cfgs, "input_dir"):
#             setattr(self._vars, "img_folder", str(getattr(self.cfgs,  "input_dir")))
#         else:
#             setattr(self._vars, "img_folder", "")

#         if hasattr(self.cfgs, "classes"):
#             if isinstance(self.cfgs.classes, list):
#                 setattr(self._vars, "classes", getattr(self.cfgs,  "classes"))
#             else:
#                 setattr(self._vars, "classes", str(getattr(self.cfgs,  "classes")))
#                 self. _vars.classes = list(self._vars.classes.split(','))
#         else:
#             setattr(self._vars, "classes", "")

#         print(">>>> vars:", self._vars)

#     def run(self, mode='train', img_exts=['bmp']):
#         imgs_info, num_data = get_images_info(mode=mode, img_folder=osp.join(self._vars.img_folder, mode), img_exts=img_exts, \
#                                 classes=self._vars.classes, roi_info=self._vars.roi_info, patch_info=self._vars.patch_info)

#         _info = []
#         num_rois_list, num_backgrounds_list = [], []
#         for img_info in imgs_info:
#             info = {}
#             info['img_file'] = img_info['img_file']
#             num_rois = len(img_info['rois'])
#             num_backgrounds = len(img_info['backgrounds'])

#             info['num_rois'] = num_rois
#             info['num_backgrounds'] = num_backgrounds

#             num_rois_list.append(num_rois)
#             num_backgrounds_list.append(num_backgrounds)

#             _info.append(info)

#         info = {'imgs_info': _info, 'total_num_imgs': len(_info), \
#                     'avg_num_rois': np.mean(num_rois_list), 'avg_num_backgrounds': np.mean(num_backgrounds_list)}

#         return info

# if __name__ == '__main__':
#     import os.path as osp
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]

#     patch_info = PatchInfo()
#     recipe = osp.join(ROOT, '../data/pie.yaml')
#     print(recipe)
#     patch_info.alg_set_cfgs(recipe=recipe)
#     print(patch_info.cfgs)
#     data = patch_info.run()
#     print(data)
