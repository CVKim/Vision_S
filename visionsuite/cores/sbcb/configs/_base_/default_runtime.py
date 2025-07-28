# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        # dict(type="TextLoggerHook", by_epoch=True),
        # dict(type="TensorboardLoggerHook"),
    ],
)
    
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
# workflow = [("train", 1)]
workflow = [("train", 1), ('val', 10)]
cudnn_benchmark = True

# Unused parameters: https://github.com/open-mmlab/mmcv/issues/1601
# enabling this seems a bit faster, which is contradicts the warnings
# find_unused_parameters = True

