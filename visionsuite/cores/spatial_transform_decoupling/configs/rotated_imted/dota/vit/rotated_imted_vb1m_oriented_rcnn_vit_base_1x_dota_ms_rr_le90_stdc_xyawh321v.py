_base_ = './rotated_imted_vb1_oriented_rcnn_vit_base_1x_dota_le90_16h.py'

model = dict(
    backbone=dict(
        use_checkpoint=False, # True, # False for A100
    ),
    roi_head=dict(
        bbox_head=dict(
            type='RotatedMAEBBoxHeadSTDC',
            dc_mode_str_list = ['', '', '', 'XY', '', 'A', '', 'WH'],
            num_convs_list   = [0, 0, 3, 3, 2, 2, 1, 1],
            am_mode_str_list = ['', '', 'V', 'V', 'V', 'V', 'V', 'V'],
            rois_mode        = 'rbbox',
            use_checkpoint=False, # True, # False for A100
        ),
    ),
)
