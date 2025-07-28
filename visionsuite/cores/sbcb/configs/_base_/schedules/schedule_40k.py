# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=1e-4)#, by_epoch=False)
# runtime settings
# runner = dict(type="IterBasedRunner", max_iters=40)
runner = dict(type="EpochBasedRunner", max_epochs=200)
checkpoint_config = dict(by_epoch=False, interval=220)
# checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=1000, metric="mIoU", pre_eval=True)
evaluation_edge = dict(interval=1000, metric="Fscore", pre_eval=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    # logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1, show=True))
