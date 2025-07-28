# dataset settings
dataset_type = "MaskDataset"
data_root = "/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/sungwoo_bottom/split_patch_mask_dataset"
classes=['background', 'scratch', 'stabbed', 'tear'],

image_size = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type="Normalize", **img_norm_cfg),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type="Normalize", **img_norm_cfg),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
    img_suffix='.bmp',
    seg_map_suffix='.bmp',
    classes=classes,
    pipeline=train_pipeline,
))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
    img_suffix='.bmp',
    seg_map_suffix='.bmp',
    classes=classes,
    pipeline=test_pipeline,
))

test_dataloader = val_dataloader


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

gpu_ids = [0, 1]  # 사용할 GPU의 ID
