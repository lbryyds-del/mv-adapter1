_base_ = '../retrieval_base_cfg.py'

ROOT  = '/home/liborui/PycharmProjects/MV-Adapter-main'
TR_SPLIT = f'{ROOT}/data/hmdb51/train-frames'
TS_SPLIT = f'{ROOT}/data/hmdb51/test-frames'
ACT_JSON = f'{ROOT}/data/hmdb51_actions.json'
TRAIN_CLASS_TXT = f'{ROOT}/data/hmdb51_classes_train.txt'
VAL_CLASS_TXT   = f'{ROOT}/data/hmdb51_classes_test.txt'

max_frames = 8
max_words  = 32

# === 测试 episodic ===
test_dataset = dict(
    type='EpisodicFramesDataset',
    split_dir=TS_SPLIT,
    label_map_json=ACT_JSON,
    class_list_file=VAL_CLASS_TXT,
    way=5,
    shot=1,                     # 评的时候要能裁 1/5，就设最大
    query_per_class=1,
    max_frames=max_frames,
    episodes_per_epoch=10000,
    eval_episodes=10000,        # 真正要测多少集
    eval_shots=[1],          # 跟 runner 对齐
)

# dataloader 还是要有的（eval 用）
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    multiprocessing_context='fork',
)

# 训练这次不用，就给一个最小的占位
train_dataset = None
train_dataloader = None

model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
    sub_act_map_path=ACT_JSON,
    precompute_text_embed=False,
    fp32=False,
    support_proto_alpha=0.5,    # 会从 ckpt 里 load 覆盖
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
                temporal=['c','p'],
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

# 不训练，所以给个空壳
optimizer = dict(
    trainable_type=[],
    warmup_proportion=0.0,
    default=dict(lr=1e-7, weight_decay=0.0),
)

# 关键：max_epochs=0 就只跑 evaluate
train_cfg = dict(
    optimizer=optimizer,
    max_epochs=0,
    val_begin=0,
    val_interval=1,
    gradient_accumulation_steps=1,
)

# 加载你最好的模型
checkpoint = 'work_dirs/hmdb51_fewshot_train/0/9e_top19329.pth'
