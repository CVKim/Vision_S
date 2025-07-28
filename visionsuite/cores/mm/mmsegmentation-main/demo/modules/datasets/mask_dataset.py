from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MaskDataset(BaseSegDataset):
    """
    In segmentation map annotation for MaskDataset, 0 stands for background, which
    is not included in defined categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('scratch', 'stabbed', 'tear'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230]])

    def __init__(self,
                 img_suffix='.bmp',
                 seg_map_suffix='.bmp',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
