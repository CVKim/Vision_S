import torch


def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors
        import v2_extras

        return torchvision.transforms.v2, torchvision.tv_tensors, v2_extras
    else:
        import visionsuite.engines.utils.torch_utils.transforms as transforms

        return transforms, None, None


class TrainTransform:
    def __init__(
        self,
        resize,
        normalize,
        backend="pil",
        use_v2=False,
    ):
        T, tv_tensors, v2_extras = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.Resize(size=(resize['height'], resize['width']))]

        if backend == "pil":
            transforms += [T.PILToTensor()]

        if use_v2:
            img_type = tv_tensors.Image if backend == "tv_tensor" else torch.Tensor
            transforms += [
                T.ToDtype(dtype={img_type: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True)
            ]
        else:
            # No need to explicitly convert masks as they're magically int64 already
            transforms += [T.ToDtype(torch.float, scale=True)]

        if normalize == 'imagenet':
            transforms += [T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
            
        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class ValTransform:
    def __init__(
        self, resize, normalize, backend="pil", use_v2=False
    ):
        T, _, _ = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.Resize(size=(resize['height'], resize['width']))]

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        if normalize == 'imagenet':
            transforms += [T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]

        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
