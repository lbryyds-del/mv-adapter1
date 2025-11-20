# dataloaders/data_dataloaders.py
import torch
from torch.utils.data import DataLoader

from registry import DATASETS


def _merge_tensor_items(items):
    """
    合并一组 tensor / 标量：
    - episodic dataset 的 batch_size=1，本来就给了一个完整的 tensor => 直接返回
    - 普通 dataset，batch_size>1，每个都是 tensor => stack
    - 一串标量 => torch.tensor(...)
    """
    if len(items) == 1:
        return items[0]

    if all(torch.is_tensor(x) for x in items):
        return torch.stack(items, dim=0)

    return torch.tensor(items)


collate_func = dict(
    stack=lambda xs: torch.stack(xs, dim=0),
    cat=lambda xs: torch.cat(xs, dim=0),
    list_plus=lambda xs: xs,
    tensor=_merge_tensor_items,
)


def collate_fn(batch_data):
    """
    期待 dataset 返回这样的结构：
    {
        "tensor": {
            "video_batch": ...,
            "v_mask": ...,
            "label": ...,
            # few-shot 时还可以带：
            # "support_video_batch": ...,
            # "support_v_mask": ...,
            # "support_label": ...,
            # "query_video_batch": ...,
            # "query_v_mask": ...,
            # "episode_class_ids": ...,
        },
        "list_plus": {...}
    }
    这里逐组 merge。
    """
    collated = {}
    for group_name in batch_data[0]:
        merge_fn = collate_func.get(group_name, _merge_tensor_items)
        group_keys = batch_data[0][group_name].keys()

        merged_group = {}
        for k in group_keys:
            values = [sample[group_name][k] for sample in batch_data]
            merged_group[k] = merge_fn(values)

        collated.update(merged_group)

    return collated


def build_loader(cfg):
    """
    支持 test-only：
    - cfg.train_dataset 允许为 None，或 max_epochs=0 时跳过训练 loader
    - 必须构建 val_dataloader 用于评估
    - 在无训练集时将 cfg.total_step 置 0
    """
    # ----------------------
    # 1) 训练（可选）
    # ----------------------
    train_dataset = None
    train_dataloader = None
    train_sampler = None

    has_train_dt = getattr(cfg, "train_dataset", None) is not None
    do_train = int(getattr(cfg, "train_cfg", {}).get("max_epochs", 0)) > 0

    if has_train_dt and do_train:
        train_dt_cfg = cfg.train_dataset.copy()
        train_dt_cls = train_dt_cfg.pop("type")
        train_dataset = DATASETS.get(train_dt_cls)(**train_dt_cfg)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        td_kwargs = {}
        if hasattr(cfg, "train_dataloader") and isinstance(cfg.train_dataloader, dict):
            td_kwargs = {k: v for k, v in cfg.train_dataloader.items() if k != "shuffle"}

        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=collate_fn,
            **td_kwargs
        )

    # ----------------------
    # 2) 验证/测试（必选）
    # ----------------------
    test_dataset_cfg = getattr(cfg, "test_dataset", None)
    if not test_dataset_cfg:
        raise RuntimeError("test_dataset 未配置，无法评估。")

    test_dt_cfg = test_dataset_cfg.copy()
    test_dt_cls = test_dt_cfg.pop("type")
    test_dataset = DATASETS.get(test_dt_cls)(**test_dt_cfg)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, shuffle=False, drop_last=False
    )
    vd_kwargs = {}
    if hasattr(cfg, "val_dataloader") and isinstance(cfg.val_dataloader, dict):
        vd_kwargs = {k: v for k, v in cfg.val_dataloader.items() if k != "shuffle"}

    val_dataloader = DataLoader(
        test_dataset,
        sampler=val_sampler,
        collate_fn=collate_fn,
        **vd_kwargs
    )

    # ----------------------
    # 3) total_step
    # ----------------------
    if train_dataloader is not None:
        cfg.total_step = int(len(train_dataloader)) * int(cfg.train_cfg.max_epochs)
    else:
        cfg.total_step = 0

    return train_dataloader, val_dataloader, train_sampler, val_sampler

