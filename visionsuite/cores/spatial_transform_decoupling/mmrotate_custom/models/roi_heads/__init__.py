# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_heads import (RotatedBBoxHead, RotatedConvFCBBoxHead,
                         RotatedShared2FCBBoxHead)
from mmrotate.models.roi_heads.gv_ratio_roi_head import GVRatioRoIHead
from mmrotate.models.roi_heads.oriented_standard_roi_head import OrientedStandardRoIHead
from mmrotate.models.roi_heads.roi_extractors import RotatedSingleRoIExtractor
from mmrotate.models.roi_heads.roi_trans_roi_head import RoITransRoIHead
from .custom_rotate_standard_roi_head import CustomRotatedStandardRoIHead

from .rotate_standard_roi_head_imted import RotatedStandardRoIHeadimTED
from .oriented_standard_roi_head_imted import OrientedStandardRoIHeadimTED

__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'CustomRotatedStandardRoIHead', 'RotatedSingleRoIExtractor',
    'OrientedStandardRoIHead', 'RoITransRoIHead', 'GVRatioRoIHead',

    'RotatedStandardRoIHeadimTED',
    'OrientedStandardRoIHeadimTED',
]
