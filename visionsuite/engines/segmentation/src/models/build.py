from visionsuite.engines.segmentation.utils.registry import MODELS

def build_model(**config):
    model = MODELS.get(config['type'], 
                       case_sensitive=config['case_sensitive'])()
    
    return model



# import torchvision 
# import os.path as osp 
# import torch 
# import torch.nn as nn 

# def get_model(args, num_classes, device):
        
#     if args['model'] == 'unet3plus':        
#         from models.unet3plus.models.UNet_3Plus import UNet_3Plus
#         model = UNet_3Plus(n_classes=num_classes)
        
#         model.to(device)
#         if args['distributed']:
#             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

#         model_without_ddp = model
#         if args['distributed']:
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']])
#             model_without_ddp = model.module

#         params_to_optimize = [
#             {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
#         ]
#     else:
#         model = torchvision.models.get_model(
#                 args['model'],
#                 weights=args['weights'],
#                 weights_backbone=args['weights_backbone'],
#                 num_classes=num_classes,
#                 aux_loss=args['aux_loss'],
#         )
        
#         model.to(device)
#         if args['distributed'] and args['distributed']['use']:
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['distributed']['gpu']])
#             model_without_ddp = model.module
#         else:
#             if torch.cuda.device_count() > 1 and len(args['train']['device_ids']) > 1:
#                 model = nn.DataParallel(model, device_ids=args['train']['device_ids'], 
#                                         output_device=args['train']['device_ids'][0])
#                 model_without_ddp = model.module
#             else:
#                 model_without_ddp = model
            
#         params_to_optimize = [
#             {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
#             {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
#         ]
#         if args['aux_loss']:
#             params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
#             params_to_optimize.append({"params": params, "lr": args['lr'] * 10})
    
#     return model, model_without_ddp, params_to_optimize