# configs/_adapter_base_fast.py
_base_ = '../retrieval_base_cfg.py'

ROOT = '/home/liborui/PycharmProjects/MV-Adapter-main'
HMDB51_ROOT = f'{ROOT}/data/hmdb51'

work_dir = 'hmdb51/mv_adapter_fast'

max_frames = 8
max_words = 32

weight_decay = 0.2
optimizer = dict(
    trainable_type=['adapter.*', '.*_no_decay'],
    warmup_proportion=0.1,
    default=dict(lr=1e-7, weight_decay=0.0),

    adapter_decay=dict(lr=5e-6, weight_decay=weight_decay),
    adapter_no_decay=dict(lr=5e-6, weight_decay=0.0),

    clip_decay=dict(lr=1e-7, weight_decay=weight_decay),
    clip_no_decay=dict(lr=1e-7, weight_decay=0.0),

    other_decay=dict(lr=1e-4, weight_decay=weight_decay),
    other_no_decay=dict(lr=1e-4, weight_decay=0.0),
)

train_cfg = dict(
    optimizer=optimizer,
    max_epochs=30,
    val_begin=0,
    val_interval=1,
    gradient_accumulation_steps=1,
    # ⭐ 给 runner 用的：一次性评多少个 episode
    eval_group_size=4,
)

train_dataset = dict(
    type='EpisodicFramesDataset',
    split_dir=f'{HMDB51_ROOT}/train-frames',
    label_map_json=f'{ROOT}/data/hmdb51_actions.json',
    alias_map_json=f'{ROOT}/data/hmdb51_label_alias.json',
    class_list_file=f'{ROOT}/data/hmdb51_classes_train.txt',
    way=5,
    shot=1,
    query_per_class=1,
    max_frames=max_frames,
)

test_dataset = dict(
    type='EpisodicFramesDataset',
    split_dir=f'{HMDB51_ROOT}/test-frames',
    label_map_json=f'{ROOT}/data/hmdb51_actions.json',
    alias_map_json=f'{ROOT}/data/hmdb51_label_alias.json',
    class_list_file=f'{ROOT}/data/hmdb51/hmdb51_classes_test.txt',
    way=5,
    shot=1,
    query_per_class=1,
    max_frames=max_frames,
)

# ⭐ 提高 worker，开持久 worker，减少 dataloader 开销
train_dataloader = dict(
    batch_size=4,
    num_workers=4,                 # 4 -> 8
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    multiprocessing_context='fork',
    persistent_workers=True,       # 训练长的时候能省不少时间
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,                 # 你原来就是 8，可以保留
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    multiprocessing_context='fork',
    persistent_workers=True,
)

# ⭐ 评估用的文本最好固定下来，避免每次都算一遍（会快）
model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
    sub_act_map_path=f'{ROOT}/data/hmdb51_actions.json',
    precompute_text_embed=True,      # 这里我改成 True，评估就不用老算文本了
    fp32=False,
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
    ),
)
