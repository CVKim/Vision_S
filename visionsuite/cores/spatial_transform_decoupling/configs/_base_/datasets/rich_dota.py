data_root = '/HDD/datasets/projects/rich/24.06.19/split_dataset_box_dota/'
angle_version = 'le90'
image_width = 768
image_height = 768
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(image_width, image_height)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(image_width, image_height),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('BOX',)
dataset_type = 'CustomDOTADataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8, # 4, # 8 for A100
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/labelTxt/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline, version=angle_version),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/labelTxt/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline, version=angle_version),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/images/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline, version=angle_version))
