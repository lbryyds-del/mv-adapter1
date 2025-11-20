import os
import cv2
import numpy as np

# ====== 你需要根据自己机器改的地方 ======
# 原始视频所在的根目录，比如你的视频实际在：
# /home/xxx/datasets/hmdb51 之类的
VIDEO_ROOT = "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/hmdb51"

# 三个划分表
SPLIT_TXT = {
    "train": "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/train_few_shot.txt",
    "val":   "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/val_few_shot.txt",
    "test":  "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/test_few_shot.txt",
}

# 抽完帧要放到哪
OUT_ROOT = "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51"
# ======================================


def make_uniform_indices(total_frames: int, k: int):
    """给定视频总帧数，均匀取 k 个下标，不够就补最后一帧"""
    if total_frames <= 0:
        return []
    if total_frames >= k:
        return np.linspace(0, total_frames - 1, k, dtype=int).tolist()
    # total_frames < k 的情况
    idxs = list(range(total_frames))
    while len(idxs) < k:
        idxs.append(total_frames - 1)  # 补最后一帧
    return idxs


def extract_8_frames(video_path: str, save_dir: str, k: int = 8):
    """
    更稳的版本：
    1. 把视频能读到的帧全部读进来
    2. 按实际读到的帧数去做等间隔采样
    这样就不会因为 CAP_PROP_FRAME_COUNT 不准导致只存 7 张
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 打不开视频: {video_path}")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        print(f"[WARN] 没读到帧: {video_path}")
        return

    idxs = make_uniform_indices(total_frames, k)

    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(idxs):
        img = frames[idx]
        out_path = os.path.join(save_dir, f"{i:05d}.jpg")
        cv2.imwrite(out_path, img)

    print(f"[OK] 抽帧成功: {video_path} -> {save_dir} (读到 {total_frames} 帧, 保存 {len(idxs)} 张)")


def process_split(split_name: str, txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for rel_path in lines:
        # txt 里有一行可能就是 'avi'，要跳过
        if rel_path.lower() == "avi":
            continue

        # 变成正常分隔符，比如 test6//videos/run/... -> test6/videos/run/...
        rel_path = rel_path.replace("//", "/").lstrip("/")

        # 只拿最后两级：类别 + 文件名
        parts = rel_path.split("/")
        if len(parts) < 2:
            print(f"[SKIP] 格式不对: {rel_path}")
            continue

        cls_name = parts[-2]   # e.g. run / brush_hair
        video_name = parts[-1] # e.g. xxx.avi

        # 按你真正的存法来拼路径
        video_path = os.path.join(VIDEO_ROOT, cls_name, video_name)

        if not os.path.exists(video_path):
            print(f"[MISS] 找不到视频文件: {video_path}")
            continue

        # 输出目录：.../hmdb51/train-frames/run/xxx/
        video_stem = os.path.splitext(video_name)[0]
        out_dir = os.path.join(OUT_ROOT, f"{split_name}-frames", cls_name, video_stem)

        extract_8_frames(video_path, out_dir, k=8)


def main():
    for split, txt in SPLIT_TXT.items():
        print(f"==== 处理 {split} 集 ====")
        process_split(split, txt)


if __name__ == "__main__":
    main()
