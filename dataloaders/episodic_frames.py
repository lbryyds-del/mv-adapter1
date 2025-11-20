# dataloaders/episodic_frames.py
from __future__ import annotations

import os
import re
import glob
import json
import random
from os import path as osp
from typing import List, Dict, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from registry import DATASETS


def _natural_key(p: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p)]


class _ToRGB(object):
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")


@DATASETS.register_module()
class EpisodicFramesDataset(Dataset):
    """
    一次采一个 episode: way * (shot + query_per_class)
    这版会额外返回 episode_class_ids，这样模型可以只对这一集的类打分，runner 就不用裁剪了。
    """

    def __init__(
        self,
        split_dir: str,
        label_map_json: str,
        alias_map_json: Optional[str] = None,
        class_list_file: Optional[str] = None,
        way: int = 5,
        shot: int = 1,
        query_per_class: int = 1,
        max_frames: int = 8,
        resolution: int = 224,
        frame_patterns: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png"),
        episodes_per_epoch: int = 10000,
        **kwargs,
    ):
        super().__init__()
        assert osp.isdir(split_dir), f"split_dir 不存在：{split_dir}"
        assert osp.isfile(label_map_json), f"label_map_json 不存在：{label_map_json}"

        self.split_dir = split_dir
        self.way = way
        self.shot = shot
        self.query_per_class = query_per_class
        self.max_frames = int(max_frames)
        self.frame_patterns = frame_patterns

        # 1) label 别名
        self.alias: Dict[str, str] = {}
        if alias_map_json and osp.isfile(alias_map_json):
            with open(alias_map_json, "r", encoding="utf-8") as f:
                self.alias = json.load(f)

        # 2) 所有 json 里的 key -> idx
        with open(label_map_json, "r", encoding="utf-8") as f:
            sub_act_raw = json.load(f)
        json_keys: List[str] = list(sub_act_raw.keys())
        self.class_to_idx: Dict[str, int] = {k: i for i, k in enumerate(json_keys)}

        # 3) 如果有类名单，就只留这几类
        if class_list_file and osp.isfile(class_list_file):
            with open(class_list_file, "r", encoding="utf-8") as f:
                target_classes = [ln.strip() for ln in f if ln.strip()]
        else:
            target_classes = None

        # 4) 只记录“每个类下面有哪些视频目录”
        self.class_to_video_dirs: Dict[str, List[str]] = {}
        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = osp.join(split_dir, cls_name)
            if not osp.isdir(cls_dir):
                continue

            json_key = self.alias.get(cls_name, cls_name)
            if json_key not in self.class_to_idx:
                continue

            if target_classes is not None and (json_key not in target_classes and cls_name not in target_classes):
                continue

            video_dirs = []
            for vid in sorted(os.listdir(cls_dir)):
                vdir = osp.join(cls_dir, vid)
                if osp.isdir(vdir):
                    video_dirs.append(vdir)

            if video_dirs:
                self.class_to_video_dirs[json_key] = video_dirs

        assert len(self.class_to_video_dirs) >= way, \
            f"可用类别数({len(self.class_to_video_dirs)}) < way({way})，请检查类名单和目录。"

        # 图像预处理
        self.transform = Compose([
            Resize(resolution, interpolation=Image.BICUBIC),
            CenterCrop(resolution),
            _ToRGB(),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        # 可配置 episode 数
        self._virt_len = int(episodes_per_epoch)

    def __len__(self):
        return self._virt_len

    def _load_video_tensor(self, video_dir: str) -> torch.Tensor:
        frame_paths: List[str] = []
        for pat in self.frame_patterns:
            frame_paths.extend(glob.glob(osp.join(video_dir, pat)))
        frame_paths = sorted(frame_paths, key=_natural_key)

        n = len(frame_paths)
        if n == 0:
            img = torch.zeros(3, 224, 224)
            return img.unsqueeze(0).repeat(self.max_frames, 1, 1, 1)
        if n >= self.max_frames:
            idxs = [round(i * (n - 1) / (self.max_frames - 1)) for i in range(self.max_frames)]
            pick = [frame_paths[i] for i in idxs]
        else:
            pick = frame_paths + [frame_paths[-1]] * (self.max_frames - n)

        imgs = []
        for p in pick:
            with Image.open(p) as im:
                imgs.append(self.transform(im))
        return torch.stack(imgs, dim=0)

    def __getitem__(self, index: int):
        # 1) 一次采 way 个类
        episode_classes = random.sample(list(self.class_to_video_dirs.keys()), self.way)

        support_videos, support_v_mask, support_labels = [], [], []
        query_videos, query_v_mask, query_labels = [], [], []

        for cls_name in episode_classes:
            video_dirs = self.class_to_video_dirs[cls_name]
            need_n = self.shot + self.query_per_class
            if len(video_dirs) >= need_n:
                chosen_dirs = random.sample(video_dirs, need_n)
            else:
                chosen_dirs = [random.choice(video_dirs) for _ in range(need_n)]

            sup_dirs = chosen_dirs[: self.shot]
            qry_dirs = chosen_dirs[self.shot:]

            # support
            for vdir in sup_dirs:
                v = self._load_video_tensor(vdir)
                support_videos.append(v)
                valid = min(len(glob.glob(osp.join(vdir, "*.jpg"))), self.max_frames)
                support_v_mask.append([1] * valid + [0] * (self.max_frames - valid))
                support_labels.append(self.class_to_idx[cls_name])

            # query
            for vdir in qry_dirs:
                v = self._load_video_tensor(vdir)
                query_videos.append(v)
                valid = min(len(glob.glob(osp.join(vdir, "*.jpg"))), self.max_frames)
                query_v_mask.append([1] * valid + [0] * (self.max_frames - valid))
                query_labels.append(self.class_to_idx[cls_name])

        support_videos = torch.stack(support_videos, dim=0)           # [S, T, 3, H, W]
        support_v_mask = torch.tensor(support_v_mask, dtype=torch.long)
        support_labels = torch.tensor(support_labels, dtype=torch.long)

        query_videos = torch.stack(query_videos, dim=0)               # [Q, T, 3, H, W]
        query_v_mask = torch.tensor(query_v_mask, dtype=torch.long)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        # ☆ 这一集用到的“全局类 id”
        episode_class_ids = torch.unique(
            torch.cat([support_labels, query_labels], dim=0)
        ).long()

        return dict(
            list_plus=dict(
                video_batch=query_videos,
                support_videos=support_videos,
            ),
            tensor=dict(
                v_mask=query_v_mask,
                support_v_mask=support_v_mask,
                label=query_labels,
                support_label=support_labels,
                episode_class_ids=episode_class_ids,   # ← 给模型/runner 用
            ),
        )
