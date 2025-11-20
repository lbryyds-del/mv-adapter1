_base_ = '../retrieval_base_cfg.py'  # 继承根目录的基础检索配置

work_dir='ufc101/mv_adapter'  # 基础工作目录

# 优化器参数（与MSRVTT保持一致或根据UFC101调整）
weight_decay = 0.2
optimizer = dict(
    trainable_type=['adapter.*', '.*_no_decay'],
    adapter_decay=dict(lr=5e-6, weight_decay=weight_decay),
    adapter_no_decay=dict(lr=5e-6),
    clip_decay=dict(weight_decay=weight_decay),
    other_decay=dict(lr=1e-4, weight_decay=weight_decay),
    other_no_decay=dict(lr=1e-4, weight_decay=weight_decay),
)

# 训练配置（轮次、验证间隔等）
train_cfg = dict(
    optimizer=optimizer,
    max_epochs=5,  # 可根据UFC101调整
    val_begin=0,
    val_interval=1,
)

# 数据集参数（使用我们之前定义的UFC101Dataset）
max_frames = 16  # UFC101视频帧数量可适当增加
max_words = 32   # 动作标签较短，足够覆盖
train_dataset = dict(
    type='UFC101Dataset',  # 替换为UFC101的数据集类
    data_root='/home/liborui/PycharmProjects/MV-Adapter-main/data/ufc101',  # UFC101根目录
    split='/home/liborui/PycharmProjects/MV-Adapter-main/data/ufc101/train-frames',
    max_frames=max_frames,
)
test_dataset = dict(
    type='UFC101Dataset',
    data_root='/home/liborui/PycharmProjects/MV-Adapter-main/data/ufc101',
    split='/home/liborui/PycharmProjects/MV-Adapter-main/data/ufc101/test-frames',
    max_frames=max_frames,
)

# 模型基础配置（复用BaselineModel）
model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
)