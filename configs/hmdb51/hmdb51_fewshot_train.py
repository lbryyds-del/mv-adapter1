_base_ = '../retrieval_base_cfg.py'

ROOT  = '/home/liborui/PycharmProjects/MV-Adapter-main'
TR_SPLIT = f'{ROOT}/data/hmdb51/train-frames'
TS_SPLIT = f'{ROOT}/data/hmdb51/val-frames'
ACT_JSON = f'{ROOT}/data/hmdb51_actions.json'
TRAIN_CLASS_TXT = f'{ROOT}/data/hmdb51/hmdb51_classes_train.txt'
VAL_CLASS_TXT   = f'{ROOT}/data/hmdb51/hmdb51_classes_val.txt'

max_frames = 8
max_words  = 32

# -------- 训练 episodic --------
train_dataset = dict(
    type='EpisodicFramesDataset',
    split_dir=TR_SPLIT,
    label_map_json=ACT_JSON,
    class_list_file=TRAIN_CLASS_TXT,
    way=5,
    shot=1,
    query_per_class=1,
    max_frames=max_frames,
    episodes_per_epoch=1000,
)

# -------- 测试 / 验证 episodic --------
test_dataset = dict(
    type='EpisodicFramesDataset',
    split_dir=TS_SPLIT,
    label_map_json=ACT_JSON,
    class_list_file=VAL_CLASS_TXT,
    way=5,
    shot=1,
    query_per_class=1,
    max_frames=max_frames,
    episodes_per_epoch=1000,
    eval_episodes=1000,
    eval_shots=[1],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    multiprocessing_context='fork',
)

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    multiprocessing_context='fork',
)

model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
    sub_act_map_path=ACT_JSON,
    precompute_text_embed=False,
    fp32=False,
    support_proto_alpha=0.5,
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
                temporal=['c'],
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

# 【重点修改】优化器配置
optimizer = dict(
    # 1. 确保 task_encoder 被列入可训练参数
    trainable_type=['adapter.*', 'task_encoder.*', '.*_no_decay'],

    warmup_proportion=0.1,

    # 默认学习率维持低位 (给未匹配到的参数)
    default=dict(lr=1e-7, weight_decay=0.0),

    # Adapter 的学习率 (通常比较小，因为是微调)
    adapter_decay=dict(lr=5e-6, weight_decay=0.2),
    adapter_no_decay=dict(lr=5e-6, weight_decay=0.0),

    # CLIP 主干的学习率 (必须非常小)
    clip_decay=dict(lr=0.0, weight_decay=0.0),
    clip_no_decay=dict(lr=0.0, weight_decay=0.0),

    # 其他参数 (通常用于 head 等)
    other_decay=dict(lr=1e-4, weight_decay=0.2),
    other_no_decay=dict(lr=1e-4, weight_decay=0.0),

    # 【新增】Task Encoder 的专属学习率配置
    # 必须给它足够大的学习率 (1e-4)，因为它是一个从零开始初始化的 MLP
    # 如果只有 1e-7，它几乎不会收敛
    task_encoder_decay=dict(lr=1e-4, weight_decay=0.2),
    task_encoder_no_decay=dict(lr=1e-4, weight_decay=0.0),
)

train_cfg = dict(
    optimizer=optimizer,
    max_epochs=20,
    val_begin=0,
    val_interval=1,
    gradient_accumulation_steps=1,
)
