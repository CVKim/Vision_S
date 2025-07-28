

## How to RUN

#### To prepare datsaet

You can refer `ultralytics/dataset_converter.py`


#### To train

```cmd
python ultralytics/train.py
```

- cfg.yaml
    ```yaml
    output_dir: /HDD/datasets/projects/rich/24.06.19/benchmark/yolo_obb
    ```

- data.yaml
    ```yaml
    path: /HDD/datasets/projects/rich/24.06.19/split_dataset_box_yolo_obb
    train: images/train
    val: images/val

    names:
    0: BOX
    ```
    You can refer [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

- train.yaml
    ```yaml
    task: obb_detection
    model_name: yolov8
    backbone: l

    epochs: 300
    batch: 8
    imgsz: 768
    device: 0,1,2,3
    lrf: 0.001
    flipud: 0.25
    fliplr: 0.25
    scale: 0.2
    degrees: 45
    ```
    You can refer [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)

