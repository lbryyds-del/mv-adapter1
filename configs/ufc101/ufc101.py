# configs/ufc101/ufc101.py
_base_ = './_adapter_base.py'

# === 数据规格 ===
max_frames = 8
max_words  = 32

# === 路径按你的机器填写 ===
UF_ROOT  = '/home/liborui/PycharmProjects/MV-Adapter-main'
ACT_JSON = f'{UF_ROOT}/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51_actions.json'
TR_SPLIT = f'{UF_ROOT}/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/train-frames'
TS_SPLIT = f'{UF_ROOT}/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/test-frames'
ALIAS_JSON = f'{UF_ROOT}/data/label_alias.json'   # 可选，若不存在会被忽略

# === 数据集（直接扫目录） ===
train_dataset = dict(
    type='UCF101FramesDataset',
    split_dir=TR_SPLIT,
    label_map_json=ACT_JSON,
    alias_map_json=ALIAS_JSON,
    max_frames=max_frames,
)
test_dataset = dict(
    type='UCF101FramesDataset',
    split_dir=TS_SPLIT,
    label_map_json=ACT_JSON,
    alias_map_json=ALIAS_JSON,
    max_frames=max_frames,
)

# === DataLoader ===
train_dataloader = dict(
    batch_size=16, num_workers=8, pin_memory=True,
    shuffle=True, drop_last=True, multiprocessing_context='fork'
)
val_dataloader = dict(
    batch_size=32, num_workers=1, pin_memory=True,
    shuffle=False, drop_last=False, multiprocessing_context='fork'
)

# === 模型 ===
model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
    sub_act_map_path=ACT_JSON,
    precompute_text_embed=True,   # 在 GPU 上预计算类别原型
    fp32=True,                    # 先用 FP32 更稳，后续可视显存改回半精度
    adapter=dict(
        visual=dict(
            mlp=dict(
                adapter_idx=list(range(12)),
                forward='seq',
                type='VideoAdapter',
                embed_dim=768,
                cls_mid=64,
                n_head=2,
                seq_len=max_frames,
                temporal=['c', 'p'],
                pca=False,
            )
        ),
        text=dict(
            mlp=dict(
                adapter_idx=list(range(12)),
                forward='seq',
                type='TextTransfAdapter',
                embed_dim=512,
                mid_dim=64,
                n_head=2,
                attn_type='uni',
                seq_len=max_words,
            )
        ),
    )
)

# === 优化器/训练配置（建议值） ===
optimizer = dict(
    trainable_type=['adapter.*','.*_no_decay'],
    warmup_proportion=0.1,
    default=dict(lr=1e-7, weight_decay=0.0),
    adapter_decay=dict(lr=1e-4, weight_decay=0.2),
    adapter_no_decay=dict(lr=1e-4, weight_decay=0.0),
    clip_decay=dict(lr=1e-7, weight_decay=0.2),
    clip_no_decay=dict(lr=1e-7, weight_decay=0.0),
    other_decay=dict(lr=1e-4, weight_decay=0.05),
    other_no_decay=dict(lr=1e-4, weight_decay=0.0),
)

train_cfg = dict(
    max_epochs=30,     # UCF101 建议 20~30 起步
    val_begin=0,
    val_interval=1,
    gradient_accumulation_steps=1
)
