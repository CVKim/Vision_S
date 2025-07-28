# dataset settings
dataset_type = "MaskDataset"
data_root = "/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/sungwoo_bottom/split_patch_mask_dataset"
classes=['scratch', 'stabbed', 'tear'],

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Normalize", **img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
]
# val_pipeline = [
#     dict(type="LoadImageFromFile"),
#     dict(type="LoadAnnotations"),
#     dict(
#         type="MultiScaleFlipAug",
#         img_scale=(512, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type="Resize", keep_ratio=True),
#             dict(type="RandomFlip"),
#             dict(type="Normalize", **img_norm_cfg),
#             dict(type="ImageToTensor", keys=["img"]),
#             dict(type="Collect", keys=["img", "gt_semantic_seg"]),
#             # dict(type="Collect", keys=["img"]),
#         ],
#     ),
# ]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images",
        ann_dir="train/masks",
        img_suffix='.bmp',
        seg_map_suffix='.bmp',
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="val/images",
        ann_dir="val/masks",
        img_suffix='.bmp',
        seg_map_suffix='.bmp',
        classes=classes,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="val/images",
        ann_dir="val/masks",
        img_suffix='.bmp',
        seg_map_suffix='.bmp',
        classes=classes,
        pipeline=test_pipeline,
    ),
)
