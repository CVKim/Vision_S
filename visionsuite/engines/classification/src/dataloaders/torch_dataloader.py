import torch
from visionsuite.engines.classification.utils.registry import DATALOADERS
from visionsuite.engines.classification.utils.augment import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate


@DATALOADERS.register()
def torch_dataloader(dataset, sampler, batch_size, workers, mixup_cutmix=None):
    
    collate_fn = default_collate
    if mixup_cutmix:
        mixup_cutmix = get_mixup_cutmix(
                mixup_alpha=mixup_cutmix['mixup_alpha'], 
                cutmix_alpha=mixup_cutmix['cutmix_alpha'], 
                num_classes=len(dataset.classes), 
                use_v2=mixup_cutmix['use_v2']
            )
        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))
            
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return dataloader

