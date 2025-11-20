
import os
import copy
import torch
from torch import nn
import torch.distributed as dist

from .modeling import BaselineModel  # 如果没用可以删
from registry import MODELS


def _load_state_dict_flex(model, ckpt_path):
    """灵活加载：兼容 state_dict/不带前缀/带 module. 前缀"""
    obj = torch.load(ckpt_path, map_location='cpu')
    state = obj.get('state_dict', obj)
    # 去掉常见前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        if k.startswith('model.'):
            k = k[len('model.'):]
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected


def init_model(cfg, device):
    # 不要就地 pop，避免 cfg.model 在其他地方还要用
    model_cfg = copy.deepcopy(cfg.model)
    model_type = model_cfg.pop('type')
    model_cls = MODELS.get(model_type)
    model = model_cls(**model_cfg)

    # 加载权重（可兼容多种 ckpt 格式）
    if getattr(cfg, 'checkpoint', None):
        missing, unexpected = _load_state_dict_flex(model, cfg.checkpoint)
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[init_model] loaded ckpt: {cfg.checkpoint}")
            if missing:
                print(f"[init_model] missing keys: {list(missing)[:8]}{' ...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"[init_model] unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")

    # 设设备
    # 若用 torchrun，会有 LOCAL_RANK；否则默认为 0
    local_rank = int(os.environ.get("LOCAL_RANK", str(getattr(cfg, 'local_rank', 0))))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model.to(device)

    # 仅在已初始化分布式时包 DDP；单卡直接返回原模型
    if dist.is_available() and dist.is_initialized():
        # 若模型所有参数在每个 step 都参与反传，设为 False 更快
        find_unused = False
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            broadcast_buffers=False,        # 通常对检索/CLIP类模型更省同步
            find_unused_parameters= True,
            static_graph=False              # 若计算图恒定且 PyTorch 版本足够新，可考虑 True
        )

    return model
