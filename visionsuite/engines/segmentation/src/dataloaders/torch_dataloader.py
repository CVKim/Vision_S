import torch
from visionsuite.engines.segmentation.utils.registry import DATALOADERS
from torch.utils.data.dataloader import default_collate


@DATALOADERS.register()
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, 
                 num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, 
                 timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None,  
                 prefetch_factor=None, persistent_workers=False, pin_memory_device=''):
        
        if collate_fn is None:
            collate_fn = default_collate
            
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, 
                 num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, 
                 timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, generator=generator,  
                 prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
            
            
    def vis(self, output_dir, **kwargs):
        if kwargs['use']:
            import numpy as np
            from visionsuite.engines.utils.functionals import denormalize
            import imgviz
            import cv2 
            import os
            import os.path as osp
            
            sampling_ratio = float(kwargs['sampling_ratio'])
            resize_ratio = float(kwargs['resize_ratio'])
            grid_rows, grid_cols = int(kwargs['grid_rows']), int(kwargs['grid_cols'])
            filename_h = int(kwargs['filename_h'])
                    
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            
            color_map=imgviz.label_colormap(256)
            origin = 25,25
            font = cv2.FONT_HERSHEY_SIMPLEX
            image_channel_order = 'rgb'
            input_channel = 3
            total_cnt = 0
            grid_cnt = 0
            for batch_idx, (batch_image, batch_target, batch_filename) in enumerate(self):
                height, width = batch_image.shape[-2:]
                
                grid_w, grid_h, grid_ch = int(width*resize_ratio), int(height*resize_ratio), 3
                grid_w *= 2
                grid_image_shape = (grid_w, grid_h)
                
                grid_h += filename_h
                if total_cnt == 0:
                    grid = np.zeros(((grid_h + filename_h)*grid_cols, grid_w*grid_rows, grid_ch))
                
                for step_idx, (image, target, filename) in enumerate(zip(batch_image, batch_target, batch_filename)):
                    
                    if sampling_ratio < np.random.rand():
                        continue
                    
                    image = image.numpy().transpose((1, 2, 0))
                    if denormalize:
                        image = denormalize(image)
                    target = color_map[target.numpy().astype(np.uint8)].astype(np.uint8)
                    if input_channel == 3:
                        if image_channel_order == 'rgb':
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        elif image_channel_order == 'bgr':
                            pass
                        else:
                            raise ValueError(f"There is no such image_channel_order({image_channel_order})")

                    elif input_channel == 1:
                        NotImplementedError(f"There is not yet training for input_channel ({input_channel})")
                    else:
                        raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

                    filename_np = np.zeros((filename_h, grid_w, input_channel), np.uint8)
                    cv2.putText(filename_np, filename, origin, font, 0.6, (255,255,255), 1)
                    target = cv2.addWeighted(image, 0.1, target, 0.9, 0)
                    it = cv2.hconcat([image, target])
                    grid_it = cv2.resize(it, grid_image_shape)
                    grid_it = cv2.vconcat([filename_np, grid_it])
                    col_start = (total_cnt // grid_cols) % grid_cols * grid_h
                    col_end = col_start + grid_h
                    row_start = (total_cnt % grid_rows) * grid_w
                    row_end = row_start + grid_w

                    if total_cnt != 0 and total_cnt%(grid_cols*grid_rows) == 0:
                        grid_cnt += 1
                        grid = np.zeros(((grid_h + filename_h)*grid_cols, grid_w*grid_rows, grid_ch))

                    grid[col_start:col_end, row_start:row_end, :] = grid_it
                    cv2.imwrite(osp.join(output_dir, f'{grid_cnt}.png'), grid)
                    
                    total_cnt += 1
