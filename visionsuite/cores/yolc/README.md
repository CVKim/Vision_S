# You Only Look Clusters for Tiny Object Detection in Aerial Images

This is the implementation of "YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images".[[Paper](https://arxiv.org/abs/2404.06180)]

<p align="center">
    <img src="framework.jpg"/>
</p>

## docker 
```cmd
docker build -t mmcv https://github.com/open-mmlab/mmcv.git#master:docker/release --build-arg MMCV=1.6.1 --build-arg PYTORCH=1.9.0 --build-arg CUDA=11.1 --build-arg CUDNN=8
```
- install
```
pip install mmdet==2.26.0
pip install kornia==0.6.9 --no-deps
```

##  Train
#### Training on a single GPU
```
python train.py configs/yolc.py
```

#### Training on multiple GPUs
```
./dist_train.sh configs/yolc.py <your_gpu_num>
```

## Citation
If you find our paper is helpful, please consider citing our paper:
```BibTeX
@article{liu2024yolc,
  title={YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images},
  author={Liu, Chenguang and Gao, Guangshuai and Huang, Ziyue and Hu, Zhenghui and Liu, Qingjie and Wang, Yunhong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```