## Set environment to run mm

1. Build docker image
```
docker build -t mmcv2.1.0 https://github.com/open-mmlab/mmcv.git#master:docker/release --build-arg PYTORCH=1.8.0 --build-arg CUDA=11.1 --build-arg CUDNN=8
```

2. Run docker & Install mmcv
```
mim install "mmcv==2.1.0"
```


> docker pull onedang2/mmcv2.1.0-torch1.8.0-cuda11.1

