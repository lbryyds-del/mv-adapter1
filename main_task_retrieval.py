# main_task_retrieval.py
from __future__ import absolute_import, division, unicode_literals, print_function

import random
import os
from os import path as osp
import argparse
import time

import torch
import numpy as np
import torch.distributed as dist
from mmengine import Config, DictAction

from dataloaders.data_dataloaders import build_loader
from optimizer import init_optimizer
from registry import RUNNERS
from modules import init_model
from util import Loggers, get_logger, my_log


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--work-root', default=None,
                        help='the dir of pretrained models and log')
    parser.add_argument('--checkpoint', default=None,
                        help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--fp32', action='store_true', help='use fp32')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config',
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    cfg.git_id = os.popen('git rev-parse HEAD').read().strip()
    cfg.git_msg = os.popen('git log --pretty=format:"%s" {} -1'.format(cfg.git_id)).read().strip()
    cfg.trial_id = os.popen('echo $ARNOLD_TRIAL_ID').read().strip()

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint

    if args.seed is not None:
        cfg.seed = args.seed

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.local_rank = args.local_rank
    cfg.work_dir = osp.join(cfg.work_dir, str(cfg.seed))
    cfg.model.fp32 = args.fp32
    if args.fp32:
        if hasattr(cfg, 'train_dataloader'):
            cfg.train_dataloader.batch_size = max(1, cfg.train_dataloader.batch_size // 2)
        if hasattr(cfg, 'val_dataloader'):
            cfg.val_dataloader.batch_size = max(1, cfg.val_dataloader.batch_size // 2)

    if args.work_root is not None:
        current_root = './'
        new_root = args.work_root
        cfg.work_dir = cfg.work_dir.replace(current_root, new_root)

        if hasattr(cfg, 'train_dataset'):
            for key in ['data_root', 'split_dir', 'label_map_json', 'alias_map_json', 'class_list_file']:
                if key in cfg.train_dataset and isinstance(cfg.train_dataset[key], str):
                    cfg.train_dataset[key] = cfg.train_dataset[key].replace(current_root, new_root)
        if hasattr(cfg, 'test_dataset'):
            for key in ['data_root', 'split_dir', 'label_map_json', 'alias_map_json', 'class_list_file']:
                if key in cfg.test_dataset and isinstance(cfg.test_dataset[key], str):
                    cfg.test_dataset[key] = cfg.test_dataset[key].replace(current_root, new_root)
        if hasattr(cfg, 'model') and 'clip_cache_dir' in cfg.model:
            cfg.model.clip_cache_dir = cfg.model.clip_cache_dir.replace(current_root, new_root)

    if dist.get_rank() == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(args.config) as cfg_file:
            open(osp.join(cfg.work_dir, f'{osp.basename(args.config)}'), 'a+').write(cfg_file.read())
    return cfg


def set_seed_logger(cfg):
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = dist.get_world_size()
    torch.cuda.set_device(cfg.local_rank)
    cfg.world_size = world_size
    cfg.rank = dist.get_rank()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = {
        'default': osp.join(cfg.work_dir, timestamp),
        'loss': osp.join(cfg.work_dir, 'loss'),
        'result': osp.join(cfg.work_dir, 'result'),
    }
    for n, l in log_file.items():
        if dist.get_rank() == 0:
            Loggers.loggers[n] = get_logger(l, n, not n == 'default')

    my_log(
        'CONFIG:\n' + '\n'.join([
            "{}: {}".format(key, cfg._cfg_dict[key])
            for key in sorted(cfg._cfg_dict.keys())
        ])
    )
    return cfg


def main():
    args = get_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg = set_seed_logger(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", cfg.local_rank)
    model = init_model(cfg, device)

    # build dataloader
    train_dataloader, val_dataloader, train_sampler, val_sampler = build_loader(cfg)

    optimizer = init_optimizer(cfg.optimizer, model, cfg.total_step)

    runner_cls = RUNNERS.get(cfg.get('runner', 'RetrievalRunner'))
    runner = runner_cls(
        cfg,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
    )
    runner.run()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
