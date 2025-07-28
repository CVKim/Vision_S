from visionsuite.engines.classification.utils.augment.presets import get_module
from .random_cut_mix import RandomCutMix
from .random_mix_up import RandomMixUp


def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_classes, use_v2):
    transforms_module = get_module(use_v2)

    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(
            transforms_module.MixUp(alpha=mixup_alpha, num_classes=num_classes)
            if use_v2
            else RandomMixUp(num_classes=num_classes, p=1.0, alpha=mixup_alpha)
        )
    if cutmix_alpha > 0:
        mixup_cutmix.append(
            transforms_module.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
            if use_v2
            else RandomCutMix(num_classes=num_classes, p=1.0, alpha=cutmix_alpha)
        )
    if not mixup_cutmix:
        return None

    return transforms_module.RandomChoice(mixup_cutmix)
