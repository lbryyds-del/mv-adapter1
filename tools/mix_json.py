# -*- coding: utf-8 -*-
import os, os.path as osp, json, glob
import argparse
from typing import List, Optional, Dict, Any
from collections import defaultdict

# =============== 通用工具函数 ===============
def uniform_sample(files: List[str], k: int, pad_short: bool = False) -> List[str]:
    """均匀采样，不足是否补齐由 pad_short 控制"""
    if len(files) == 0:
        return []
    if len(files) >= k:
        idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)]
        return [files[i] for i in idxs]
    return files if not pad_short else files + [files[-1]] * (k - len(files))

def list_frame_files(vid_dir_abs: str) -> List[str]:
    return sorted(
        glob.glob(osp.join(vid_dir_abs, "*.jpg")) +
        glob.glob(osp.join(vid_dir_abs, "*.jpeg")) +
        glob.glob(osp.join(vid_dir_abs, "*.png"))
    )

def stem(x: str) -> str:
    return osp.splitext(osp.basename(str(x)))[0]

def try_match_dir(root: str, vid: str) -> Optional[str]:
    candidates = [vid, stem(vid)]
    if str(vid).startswith("video") and str(vid)[5:].isdigit():
        candidates.append(str(vid)[5:])
    if str(vid).isdigit():
        candidates.append(f"video{vid}")
    for name in candidates:
        p = osp.join(root, name)
        if osp.isdir(p): return p
    for name in candidates:
        hits = glob.glob(osp.join(root, "**", name), recursive=True)
        hits = [h for h in hits if osp.isdir(h)]
        if hits:
            hits.sort(key=len)
            return hits[0]
    return None

def ensure_dir(path: str):
    os.makedirs(osp.dirname(path), exist_ok=True)


# =============== 数据加载 ===============
def load_train9k_multicap_json(src_json: str) -> List[Dict[str, Any]]:
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "train_9k JSON 顶层应为数组"
    return data

def get_vid(it: Dict[str, Any]) -> Optional[str]:
    return it.get("vid") or it.get("video_id") or it.get("video") or (str(it.get("id")) if it.get("id") is not None else None)

def get_caps(it: Dict[str, Any]) -> List[str]:
    caps = it.get("caption") or it.get("sentences") or it.get("texts")
    if isinstance(caps, list):
        return [c for c in caps if isinstance(c, str) and c.strip()]
    if isinstance(caps, str):
        return [caps] if caps.strip() else []
    return []

def load_msrvtt_official_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    videos, sentences = full_data["videos"], full_data["sentences"]
    video_ids = [v["video_id"] for v in videos if "video_id" in v]
    vid_to_captions = defaultdict(list)
    for s in sentences:
        vid, cap = s.get("video_id"), s.get("caption")
        if vid and cap:
            vid_to_captions[vid].append(cap)
    return video_ids, vid_to_captions


# =============== 构建两种模式 ===============
def build_from_frames(data_root, frames_root_abs, src_json, n_frames=12, per_caption=True, pad_short=False):
    """模式A：train_9k + 已抽帧目录"""
    raw = load_train9k_multicap_json(src_json)
    out = []
    miss_dir = zero_frames = total_caps = 0

    for it in raw:
        vid_raw = get_vid(it)
        captions = get_caps(it)
        if not vid_raw or not captions:
            continue
        total_caps += len(captions)

        vid_dir = try_match_dir(frames_root_abs, str(vid_raw).replace(".mp4", ""))
        if vid_dir is None:
            miss_dir += 1
            continue

        files = list_frame_files(vid_dir)
        if not files:
            zero_frames += 1
            continue

        sampled = uniform_sample(files, n_frames, pad_short)
        frames_rel = [osp.relpath(p, data_root) for p in sampled]
        vid_final = stem(vid_dir)

        if per_caption:
            for cap in captions:
                out.append({"vid": vid_final, "caption": cap.strip(), "frames": frames_rel})
        else:
            out.append({"vid": vid_final, "caption": [c.strip() for c in captions], "frames": frames_rel})

    print(f"=== from_frames ===\n总字幕:{total_caps} 成功:{len(out)} 缺目录:{miss_dir} 空目录:{zero_frames}")
    return out


def build_from_original(json_path, frame_root_rel="frames", n_frames=12, per_caption=False,
                        frame_root_abs=None, check_frames=False, pad_short=False):
    """模式B：官方JSON + 抽帧路径"""
    vids, vid2caps = load_msrvtt_official_json(json_path)
    out, sample_id = [], 0

    for vid in vids:
        caps = vid2caps.get(vid, [])
        if not caps:
            continue

        if frame_root_abs:
            dir_abs = osp.join(frame_root_abs, vid)
            files = list_frame_files(dir_abs)
            if check_frames and not files:
                continue
            if files:
                sampled = uniform_sample(files, n_frames, pad_short)
                frames_rel = [osp.join(frame_root_rel, vid, osp.basename(p)) for p in sampled]
            else:
                frames_rel = [osp.join(frame_root_rel, vid, f"{i:04d}.jpg") for i in range(1, n_frames + 1)]
        else:
            frames_rel = [osp.join(frame_root_rel, vid, f"{i:04d}.jpg") for i in range(1, n_frames + 1)]

        if per_caption:
            for cap in caps:
                out.append({"vid": vid, "id": sample_id, "frames": frames_rel, "caption": cap.strip()})
                sample_id += 1
        else:
            out.append({"vid": vid, "id": sample_id, "frames": frames_rel,
                        "caption": [c.strip() for c in caps] if len(caps) > 1 else caps[0].strip()})
            sample_id += 1

    print(f"=== from_original ===\n视频数:{len(vids)} 输出样本:{len(out)}")
    return out


# =============== 主入口 ===============
if __name__ == "__main__":
    # 这里的路径请自行修改 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    MODE = "from_frames"   # 选 "from_frames" 或 "from_original"

    if MODE == "from_frames":
        DATA_ROOT = "/home/liborui/PycharmProjects/MV-Adapter-main/data"
        FRAMES_DIR = "/data/msvd/frames"
        SRC_JSON = "/split/msvd_train.json"
        DST_JSON = "/split/msvd_train1.json"
        result = build_from_frames(DATA_ROOT, FRAMES_DIR, SRC_JSON, n_frames=12,
                                   per_caption=True, pad_short=False)
        ensure_dir(DST_JSON)
        json.dump(result, open(DST_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"已输出：{DST_JSON}")

    else:  # from_original
        INPUT_JSON = "train_videodatainfo.json路径"
        OUTPUT_JSON = "输出json路径"
        FRAME_ROOT_ABS = "真实帧路径(可为空)"
        result = build_from_original(INPUT_JSON, frame_root_abs=FRAME_ROOT_ABS,
                                     n_frames=12, per_caption=False,
                                     check_frames=False, pad_short=False)
        ensure_dir(OUTPUT_JSON)
        json.dump(result, open(OUTPUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"已输出：{OUTPUT_JSON}")
