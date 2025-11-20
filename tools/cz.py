# uniform_extract_8frames.py
from pathlib import Path
from typing import List
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
from tqdm import tqdm

# ========= 按需修改这几个常量 =========
SRC_DIR = Path("/data/ufc101/train")     # 源视频根目录（可含类别子目录）
OUT_DIR = Path("/data/ufc101/train-frames")     # 输出根目录（不再额外加 'frames' 目录）
KEEP_CLASS_SUBDIR = True                   # True: 输出 <OUT>/<Class>/<VideoID>；False: <OUT>/<VideoID>
IMG_FORMAT = "jpg"                         # jpg / png / jpeg
IMG_QUALITY = 95                           # 仅 jpg/jpeg 生效：0~100
TARGET_FRAMES = 8                          # 固定均匀抽取 8 帧
OVERWRITE = False                          # 已存在则跳过；设为 True 会覆盖
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}
# =====================================

def list_videos(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix in VIDEO_EXTS]

def build_out_dir(video_path: Path) -> Path:
    vid_stem = video_path.stem
    if KEEP_CLASS_SUBDIR:
        cls = video_path.parent.name
        return OUT_DIR / cls / vid_stem
    else:
        return OUT_DIR / vid_stem

def save_frame(arr: np.ndarray, save_path: Path):
    img = Image.fromarray(arr)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if IMG_FORMAT.lower() in ["jpg", "jpeg"]:
        img.save(save_path, format="JPEG", quality=IMG_QUALITY, subsampling=2, optimize=True)
    else:
        img.save(save_path)

def uniform_indices(num_frames: int, k: int) -> np.ndarray:
    """返回长度为 k 的均匀采样索引；当总帧数不足 k 时，用最后一帧补齐。"""
    if num_frames <= 0:
        return np.array([], dtype=int)
    if num_frames >= k:
        return np.linspace(0, num_frames - 1, num=k, dtype=int)
    # 帧数不够：先取所有唯一帧，再用最后一帧填充到 k
    base = np.linspace(0, num_frames - 1, num=num_frames, dtype=int)
    pad = np.full((k - num_frames,), base[-1], dtype=int)
    return np.concatenate([base, pad], axis=0)

def extract_one(video_path: Path):
    out_dir = build_out_dir(video_path)
    # 如果已存在并且不覆盖，直接跳过
    if not OVERWRITE and out_dir.exists() and len(list(out_dir.glob(f"*.{IMG_FORMAT}"))) >= TARGET_FRAMES:
        return

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        n = len(vr)
        idxs = uniform_indices(n, TARGET_FRAMES)
        if idxs.size == 0:
            print(f"[WARN] 无法读取帧：{video_path}")
            return
        frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3), uint8
        for i, frame in enumerate(frames, start=1):
            save_path = out_dir / f"{i:04d}.{IMG_FORMAT}"
            save_frame(frame, save_path)
    except Exception as e:
        print(f"[ERROR] 处理失败：{video_path} -> {e}")

def main():
    videos = list_videos(SRC_DIR)
    if not videos:
        print(f"[INFO] 在 {SRC_DIR} 未找到视频文件")
        return
    for vp in tqdm(videos, desc="Extracting 8 frames/video"):
        extract_one(vp)
    print(f"[DONE] 已输出到：{OUT_DIR}")

if __name__ == "__main__":
    main()
