import json
import os.path as osp
import random
from copy import deepcopy
from glob import glob
from threading import Thread

import numpy as np

from visionsuite.engines.data.slicer.src.base_slicer import BaseSlicer
from visionsuite.engines.data.slicer.src.make_patches import get_imgs_info_from_patches


class Slicer(BaseSlicer):
    """
    ### Attributes
        imgs_info: [
                        {
                            'img_file' : image file name, # string
                            'patches': [
                                        [x1, y1, x2, y2],
                                        [x1, y1, x2, y2],
                                        ...
                                    ], # list of lists: patch coordination
                            'counts': [0, 0, ...], # list
                            'backgrounds': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...], # list of lists
                            'labels': [
                                        [{'label': 'abc', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]},
                                        {'label': '', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]}, ...],
                                        [{'label': '', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]},
                                        {'label': '', 'points': [[x1, y1], [x2, y2], ...], 'shape_type': '', 'roi': [x1, y1, x2, y2]}, ...],
                                        ...
                                    ] # list of lists: label information in each patch coordination
                        },
                        {
                            'img_file' : image file name,
                            'patches': ...,
                            ...
                        },
                        ...
                    ]
            - Each patch coordinate is synched to each labels list and count.
            - Backgrounds exist only when using patch-based training
    """

    def __init__(
        self,
        mode,
        input_dir,
        img_exts=["bmp", "png", "jpg"],
        classes=None,
        roi_info=None,
        roi_from_json=False,
        patch_info=None,
        logger=None,
    ):
        super().__init__(__class__.__name__)

        self.mode = mode
        self.input_dir = input_dir
        self.img_exts = img_exts
        self.classes = classes
        self.roi_info = roi_info
        self.roi_from_json = roi_from_json
        self.patch_info = patch_info

        if logger is not None:
            self.set_log(
                log_dir=logger.log_dir,
                log_stream_level=logger.log_stream_level,
                log_file_level=logger.log_file_level,
            )

            self._logger.info(
                f"Logger for {__class__.__name__} has been set",
                self.__init__.__name__,
                __class__.__name__,
            )
            self._logger.info(
                f"****************************************************************************************"
            )
            self._logger.info(
                f"Slicer works on {mode} at {input_dir} within {img_exts} images for {classes}"
            )
            self._logger.info(f"RoI information: {roi_info}")
            self._logger.info(f"Patch information: {patch_info}")
            self._logger.info(
                f"****************************************************************************************"
            )
        else:
            self._logger = None

        self._img_files = []
        self._set_img_files()

        self.num_data = 0

    @property
    def num_img_files(self):
        return len(self._img_files)

    def run(self):
        for img_file in self._img_files:
            self.__img_info = {
                "img_file": img_file,
                "patches": [],
                "backgrounds": [],
                "labels": [],
            }
            self._set_roi_info(img_file)
            self._slice(img_file)

            if (
                self.__img_info["patches"] is not None
                and len(self.__img_info["patches"]) == 0
            ):
                print(f"*** There is zero patches for {img_file}")
                continue

            self.imgs_info = deepcopy(self.__img_info)

    def _slice(self, img_file):
        if self.patch_info is None:
            self._slice_wo_patch_info()
        else:
            self._slice_w_patch_info(img_file)

    def _slice_wo_patch_info(self):
        if self.roi_info is None:  # w/ roi & wo/ patch
            self.__img_info["patches"] = None
            self.__img_info["backgrounds"] = None
            self.__img_info["counts"] = [0]
            self.num_data += 1
        else:  # w/ roi & wo/ patch
            self.__img_info["backgrounds"] = None
            for roi in self.roi_info:
                self.__img_info["patches"].append(roi)
                self.num_data += 1
            self.__img_info["counts"] = [0] * len(self.__img_info["patches"])

    def _slice_w_patch_info(self, img_file):
        for roi in self.roi_info:
            patches, _num_data, background_patches, labels = get_imgs_info_from_patches(
                mode=self.mode,
                img_file=img_file,
                classes=self.classes,
                patch_info=self.patch_info,
                roi=roi,
                logger=self._logger,
            )
            self.__img_info["patches"] += patches
            self.__img_info["backgrounds"] += background_patches
            self.__img_info["labels"] += labels
            self.__img_info["counts"] = [0] * len(self.__img_info["patches"])
            self.num_data += _num_data

    def _set_img_files(self):
        if self._logger is not None:
            self._logger.assertion_log(
                self.mode is not None
                and self.input_dir is not None
                and self.img_exts != [],
                RuntimeError(
                    f"At least, one of mode({self.mode}), input-dir({self.input_dir}), and extension({self.img_exts}) is not defined"
                ),
                parent_class=__class__.__name__,
                parent_fn=self._set_img_files.__name__,
            )
        for ext in self.img_exts:
            if self.mode != "test":
                self._img_files += glob(
                    osp.join(self.input_dir, self.mode, "*.{}".format(ext))
                )
            else:
                self._img_files += glob(osp.join(self.input_dir, "*.{}".format(ext)))

        if self._logger is not None:
            self._logger.info(
                f"There are {len(self._img_files)} image files with {self.img_exts} in {self.mode} mode"
            )

    def _set_roi_info(self, img_file):
        with open(osp.splitext(img_file)[0] + ".json") as f:
            _json = json.load(f)
            if (
                self.roi_from_json
                and "rois" in _json.keys()
                and len(_json["rois"]) != 0
            ):
                self.roi_info = _json["rois"]
            elif self.roi_info is None:
                self.roi_info = [[]]

    def get_patch_information(self):
        _info = []
        num_patches_list, num_backgrounds_list = [], []
        for img_info in self.imgs_info:
            info = {}
            info["img_file"] = img_info["img_file"]
            num_patches = len(img_info["patches"])
            num_backgrounds = len(img_info["backgrounds"])

            info["num_patches"] = num_patches
            info["num_backgrounds"] = num_backgrounds

            num_patches_list.append(num_patches)
            num_backgrounds_list.append(num_backgrounds)

            _info.append(info)

        info = {
            "imgs_info": _info,
            "total_num_imgs": len(_info),
            "avg_num_patches_per_image": np.mean(num_patches_list),
            "avg_num_backgrounds_per_image": np.mean(num_backgrounds_list),
        }

        return info

    def save_imgs_info(self, output_dir):
        def save_as_json():
            with open(
                osp.join(output_dir, f"{self.mode}_imgs_info.json"), "w"
            ) as json_file:
                json.dump(self.imgs_info, json_file)

        Thread(target=save_as_json).start()

    def get_classes_info_by_patch(self, patch_ratio_by_class=None):

        def _labels_to_image_weights(class_counts, nc, class_weights):
            image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
            # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
            return image_weights

        class2idx = {
            _class.lower(): int(idx) + 1 for idx, _class in enumerate(self.classes)
        }
        patch_ratio_by_class = (
            np.ones(len(self.classes))
            if patch_ratio_by_class is None
            else patch_ratio_by_class
        )
        classes_info_by_patch = {
            "labels_in_patch": [],
            "labels_in_image": [],
            "positive_point": [],
            "cnt_class_by_patch": np.zeros(len(self.classes)),
        }
        num_background_patch_by_image, total_num_patch_by_image = 0, 0
        for idx, img_info in enumerate(self.imgs_info):
            num_background_patch_by_image += len(img_info["backgrounds"])
            total_num_patch_by_image += len(img_info["backgrounds"]) + len(
                img_info["patches"]
            )
            labels_in_image = [0] * len(self.classes)
            for patches, labels in zip(img_info["patches"], img_info["labels"]):
                labels_in_patch = [0] * len(self.classes)
                for label in labels:
                    if label["label"].lower() in class2idx.keys():
                        labels_in_image[class2idx[label["label"]] - 1] += 1
                    if len(label["points"]) == 0 and label["label"] == "":
                        classes_info_by_patch["positive_point"].append(idx)
                    else:
                        if label["label"] not in classes_info_by_patch:
                            classes_info_by_patch[label["label"]] = [idx]
                        else:
                            classes_info_by_patch[label["label"]].append(idx)
                classes_info_by_patch["labels_in_patch"].append(labels_in_patch)
            classes_info_by_patch["labels_in_image"].append(labels_in_image)

        _tmp_dict = {"total_num_instance": 0}
        for key, val in classes_info_by_patch.items():
            if key in class2idx.keys():
                _tmp_dict["num_" + key + "_instance"] = len(val)
                _tmp_dict["total_num_instance"] += len(val)
                _tmp_dict["num_" + key + "_image"] = len(set(val))
                _tmp_dict["max_" + key + "_instance"] = (
                    len(val) * patch_ratio_by_class[int(class2idx[key]) - 1]
                )
        _tmp_dict["num_positive_point_instance"] = len(
            classes_info_by_patch["positive_point"]
        )
        _tmp_dict["num_positive_point_image"] = len(
            set(classes_info_by_patch["positive_point"])
        )

        classes_info_by_patch.update(_tmp_dict)
        del _tmp_dict

        classes_info_by_patch["total_num_images"] = len(self.imgs_info)
        classes_info_by_patch["num_patches_per_image"] = int(
            total_num_patch_by_image / len(self.imgs_info)
        )
        classes_info_by_patch["avg_num_backgrounds_per_image"] = (
            num_background_patch_by_image / len(self.imgs_info)
        )

        # TODO
        # class-weights
        classes_info_by_patch["class_weights"] = [
            len(classes_info_by_patch[key])
            / classes_info_by_patch["total_num_instance"]
            for key in class2idx.keys()
        ]

        # image-weights
        classes_info_by_patch["image_weights"] = _labels_to_image_weights(
            classes_info_by_patch["labels_in_image"],
            len(self.classes),
            np.ones(len(self.classes))
            - np.array(classes_info_by_patch["class_weights"]),
        )
        classes_info_by_patch["image_indices"] = random.choices(
            range(len(self.imgs_info)),
            weights=classes_info_by_patch["image_weights"],
            k=len(self.imgs_info),
        )

        # patch-weights
        classes_info_by_patch["patch_weights"] = _labels_to_image_weights(
            classes_info_by_patch["labels_in_image"],
            len(self.classes),
            np.array(classes_info_by_patch["class_weights"]),
        )
        classes_info_by_patch["patch_indices"] = random.choices(
            range(len(self.imgs_info)),
            weights=classes_info_by_patch["patch_weights"],
            k=len(self.imgs_info),
        )

        return classes_info_by_patch

    def debug(self):
        pass
