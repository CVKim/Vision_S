docker build -t mmcv https://github.com/open-mmlab/mmcv.git#main:docker/release

# docker build -t mmcv -f docker/release/Dockerfile \
#     --build-arg PYTORCH=1.11.0 \
#     --build-arg CUDA=11.3 \
#     --build-arg CUDNN=8 \
#     --build-arg MMCV=2.0.0 .