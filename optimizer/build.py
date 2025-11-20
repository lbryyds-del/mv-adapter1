import re
import torch
from .bert_adam import BertAdam
from util import my_log, overwrite


def init_optimizer(cfg, model, total_step):
    """
    cfg: cfg.optimizer
    返回:
        - BertAdam 优化器；若无可训练参数则返回 None（test-only）
    """
    if hasattr(model, 'module'):
        model = model.module

    # -------- 参数分组 --------
    params = {
        'adapter_decay': [],
        'adapter_no_decay': [],
        'clip_decay': [],
        'clip_no_decay': [],
        'other_decay': [],
        'other_no_decay': [],
    }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    for n, p in list(model.named_parameters()):
        if 'adapter' in n or 'prompt' in n or 'factor' in n:
            if any(nd in n for nd in no_decay):
                params['adapter_no_decay'].append(p)
            else:
                params['adapter_decay'].append(p)
        elif 'clip.' in n:
            if any(nd in n for nd in no_decay):
                params['clip_no_decay'].append(p)
            else:
                params['clip_decay'].append(p)
        else:
            if any(nd in n for nd in no_decay):
                params['other_no_decay'].append(p)
            else:
                params['other_decay'].append(p)

    # 允许通过分组名进行筛选（保持你原有逻辑）
    trainable_type = cfg.get('trainable_type', ['.*'])
    cfg.trainable_type = [k for k in params if any(re.match(f'{t}$', k) for t in trainable_type)]

    # 不在 trainable_type 的分组一律冻结
    for n, plist in params.items():
        if n not in cfg.trainable_type:
            for pp in plist:
                pp.requires_grad = False

    # 统计与日志
    trainable_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    clip_params   = sum(p.numel() for p in (params['clip_decay'] + params['clip_no_decay']))
    clip_nd_params = sum(p.numel() for p in params['clip_no_decay'])
    clip_d_params  = sum(p.numel() for p in params['clip_decay'])
    other_params   = sum(p.numel() for p in (params['other_decay'] + params['other_no_decay']))
    adapter_params = sum(p.numel() for p in (params['adapter_decay'] + params['adapter_no_decay']))
    total_params   = clip_params + other_params + adapter_params
    # 用全部参数做分母，更稳；加除零保护
    trainable_ratio = (trainable_params / max(total_params, 1)) * 100.0

    param_log = (
        '\nclip params: {}\n\tclip decay params: {}\n\t'
        'clip no decay params: {}, \nadapter params: {}\n'
        'other params: {}\n'
        'total params: {}\n'
        'trainable params: {}, ratio: {:.3f}%'
    )
    my_log(param_log.format(
        clip_params, clip_d_params, clip_nd_params,
        adapter_params, other_params, total_params,
        trainable_params, trainable_ratio
    ))

    # -------- 若无可训练参数 → 直接返回 None（test-only）--------
    if trainable_params == 0:
        my_log("[init_optimizer] no trainable parameters detected; return None (test-only).")
        return None

    # -------- 组建优化器参数组（跳过为空的分组）--------
    default = dict(weight_decay=0.0, lr=1e-7)
    default = overwrite(default, cfg.get('default', {}))

    optimizer_grouped_parameters = []
    for t in cfg.trainable_type:
        group_params = params.get(t, [])
        if not group_params:  # 跳过空分组，避免空 param_group
            continue
        group_cfg = overwrite(default, cfg.get(t, {}))
        optimizer_grouped_parameters.append({'params': group_params, **group_cfg})

    # 如果最终还是没有任何组，就返回 None
    if len(optimizer_grouped_parameters) == 0:
        my_log("[init_optimizer] grouped parameter list empty; return None (test-only).")
        return None

    # -------- 构建优化器 --------
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=default['lr'],
        warmup=cfg.get('warmup_proportion', 0.1),
        schedule='warmup_cosine',
        b1=0.9, b2=0.98, e=1e-6, max_grad_norm=1.0,
        t_total=total_step,
        weight_decay=default.get('weight_decay', 0.0)
    )

    return optimizer
