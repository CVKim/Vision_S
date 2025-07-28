_base_ = [
    "../_base_/models/deeplabv3plus_r50b-d8.py",
    "../_base_/datasets/mask.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
]
model = dict(auxiliary_head=None)
