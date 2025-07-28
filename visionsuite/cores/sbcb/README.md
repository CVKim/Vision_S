# [Boosting Semantic Segmentation with Semantic Boundaries](http://arxiv.org/abs/2304.09427)

### How to run

#### Set the environment

1. Download docker image
  ```
  docker build -t mmcv1.6.1-torch1.8.1-cuda11.1-cudnn8 https://github.com/open-mmlab/mmcv.git#master:docker/release --build-arg MMCV=1.6.1 --build-arg PYTORCH=1.8.1 --build-arg CUDA=11.1 --build-arg CUDNN=8
  ```

2. Run docker image

3. Install libraries
  ```
  pip install mmsegmentation==0.27.0 scikit-image scikit-learn pyEdgeEval yapf==0.40.1
  ```


#### Train

- Distributed

  ```
    bash dist_train.sh <number of gpus to use>
  ```

- Not distributed

  ```
  python train.py
  ```
