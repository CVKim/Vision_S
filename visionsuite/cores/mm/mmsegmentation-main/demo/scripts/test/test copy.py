# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.customs.datasets import MaskDataset

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('--config', default=ROOT / 'configs/models/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-20k_sungwoo_bottom-512x512.py')
    parser.add_argument('--checkpoint', default='/HDD/etc/outputs/sungwoo_bottom/train/deeplabv3plus/iter_2200.pth')
    parser.add_argument(
        '--work-dir', default='/HDD/etc/outputs/sungwoo_bottom/test/deeplabv3plus',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir', default='/HDD/etc/outputs/sungwoo_bottom/test/deeplabv3plus',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.gpu_ids = [1]
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # if args.show or args.show_dir:
    #     cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    import torch 
    import imgviz
    import numpy as np
    import cv2

    color_map = imgviz.label_colormap(255)
    runner.model.to('cuda:1')
    runner.model.eval()

    for idx, batch in enumerate(runner.test_loop.dataloader):
        with torch.no_grad():
            outputs = runner.model.test_step(batch)
            
        for output in outputs:
            original_img = cv2.imread(output.img_path) 
            gt_img = color_map[cv2.imread(output.seg_map_path, 0)]
            vis_gt = cv2.addWeighted(original_img, 0.4, gt_img, 0.6, 0)
            
            filename = osp.split(osp.splitext(output.img_path)[0])[-1]
            h, w = output.ori_shape
            vis_img = np.zeros((h, w*2, 3))
            gt_sem_seg = output.gt_sem_seg.data.cpu().detach().numpy().squeeze(0).astype(np.uint8)
            pred_sem_seg = output.pred_sem_seg.data.cpu().detach().numpy().squeeze(0).astype(np.uint8)
            seg_logits = output.seg_logits.data.cpu().detach().numpy()
            vis_img[:, :w, :] = vis_gt
            vis_img[:, w:, :] = cv2.addWeighted(original_img, 0.4, color_map[pred_sem_seg], 0.6, 0)
            
            cv2.imwrite(osp.join(args.show_dir, filename + '.png'), vis_img)
            
            
if __name__ == '__main__':
    main()
