# modules/modeling.py
from __future__ import absolute_import, division, print_function

import json
import logging
import os
import os.path as osp
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from mmengine.runner.amp import autocast

import clip
from modules.tokenization_clip import SimpleTokenizer
from modules.module_clip import CLIP, convert_weights
from registry import MODELS

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "CLS_TOKEN": "<|startoftext|>",
    "SEP_TOKEN": "<|endoftext|>",
    "MASK_TOKEN": "[MASK]",
    "UNK_TOKEN": "[UNK]",
    "PAD_TOKEN": "[PAD]",
}

_PT_NAME_OFFICIAL = {
    "ViT-B/32": "ViT-B/32",
    "v32": "ViT-B/32",
    "ViT-B/16": "ViT-B/16",
    "v16": "ViT-B/16",
    "ViT-L/14": "ViT-L/14",
    "vl14": "ViT-L/14",
}


# ==================================================================
# 【核心修改】任务上下文编码器 (Zero Init + Tanh)
# ==================================================================
class TaskContextEncoder(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=64, hidden_dim=256):
        """
        用于将 Support Set 的特征压缩成 Adapter 的调节参数 Sigma (Delta)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(hidden_dim, bottleneck_dim)
            # 注意：这里去掉了 Sigmoid，改在 forward 里处理
        )

        # 【关键策略 1: 零初始化】
        # 初始化最后一层为全 0，确保初始状态下 Sigma=0
        # 这样初始效果完全等同于 Baseline，解决了冷启动掉点问题
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, support_feats):
        """
        support_feats: [N_support, input_dim]
        return: [bottleneck_dim]
        """
        # 1. 聚合: 全局平均池化
        global_context = torch.mean(support_feats, dim=0)  # [input_dim]

        # 2. 生成 Delta 量
        # 使用 Tanh 将变化限制在 [-1, 1] 之间，再乘以 0.5
        # 意味着 Adapter 的通道权重最多 增加50% 或 减少50%
        # 这是一个安全的“微调”范围
        sigma = torch.tanh(self.mlp(global_context)) * 0.5

        return sigma


@MODELS.register_module()
class BaselineModel(nn.Module):
    """
    融入 AMAL 双层优化的 Episodic 模型
    """

    def __init__(
            self,
            clip_arch: str = "v16",
            adapter: Optional[dict] = None,
            checkpoint: Optional[str] = None,
            max_words: int = 32,
            sub_act_map_path: str = "data/action_sub_acts.json",
            sub_act_map: Optional[dict] = None,
            precompute_text_embed: bool = False,
            vis_chunk_size: int = 32,
            fp32: bool = False,
            support_proto_alpha: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        adapter = adapter or {}
        self.tokenizer = SimpleTokenizer("modules/bpe_simple_vocab_16e6.txt.gz")
        self.cache_dir = kwargs.get("clip_cache_dir", ".cache/clip")
        os.makedirs(os.path.expanduser(self.cache_dir), exist_ok=True)

        self.epoch = 0
        self.checkpoint = checkpoint
        self.max_words = max_words
        self.fp32 = fp32
        self.vis_chunk_size = vis_chunk_size
        self.support_proto_alpha = float(support_proto_alpha)

        # 1) CLIP
        self.load_clip(clip_arch, adapter)

        # 精度
        self.training_type = torch.float32 if self.fp32 else torch.float16
        if self.fp32:
            self.clip.float()

        # 2) 【新增】初始化任务编码器
        visual_dim = self.clip.visual.output_dim  # 768
        # 获取 Adapter 中间层维度，默认为 64
        adapter_mid_dim = adapter.get('visual', {}).get('mlp', {}).get('cls_mid', 64)

        self.task_encoder = TaskContextEncoder(
            input_dim=visual_dim,
            bottleneck_dim=adapter_mid_dim
        ).float()  # 确保精度匹配

        # 类别描述 / 文本子动作映射
        if sub_act_map is not None and isinstance(sub_act_map, dict):
            self.sub_act_map = sub_act_map
        elif sub_act_map_path and osp.isfile(sub_act_map_path):
            with open(sub_act_map_path, "r", encoding="utf-8") as f:
                self.sub_act_map = json.load(f)
        else:
            self.sub_act_map = None
            logger.warning("[BaselineModel] 未找到 sub_act_map，文本原型将无法构建。")

        self.all_labels = list(self.sub_act_map.keys()) if self.sub_act_map else []

        self.register_buffer("text_embeddings", None, persistent=False)
        self._want_precompute = False
        self._did_precompute = False

    # ------------------------------------------------------------------
    # AMAL 辅助函数：参数注入与重置
    # ------------------------------------------------------------------
    def inject_task_parameters(self, sigma):
        """将生成的 Sigma 注入到所有 VideoAdapter 中"""
        for name, module in self.clip.visual.named_modules():
            # 寻找 VideoAdapter 并注入
            if hasattr(module, 'set_scale'):
                module.set_scale(sigma)

    def reset_task_parameters(self):
        """清空 Adapter 的动态参数"""
        for name, module in self.clip.visual.named_modules():
            if hasattr(module, 'set_scale'):
                module.set_scale(None)

    # ------------------------------------------------------------------
    # CLIP 加载 (保持不变)
    # ------------------------------------------------------------------
    def load_clip(self, clip_arch, adapter):
        state_dict = {}
        official_arch = _PT_NAME_OFFICIAL[clip_arch]
        clip_model, _ = clip.load(
            official_arch,
            device="cpu",
            download_root=os.path.expanduser(self.cache_dir),
        )
        clip_model.eval()
        clip_state_dict = clip_model.state_dict()

        attn_pattern = r"resblocks\.[0-9]+\.attn.in_proj"
        ln1_pattern = r"resblocks\.[0-9]+\.ln_1"
        ln2_pattern = r"resblocks\.[0-9]+\.ln_2"
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if re.search(attn_pattern, new_key):
                qkv = val.reshape(3, -1, *val.shape[1:]).clone()
                for i, name in enumerate(["q", "k", "v"]):
                    state_dict[new_key.replace("attn.in_proj_", f"attn.{name}_proj.")] = qkv[i]
                continue
            elif re.findall(ln1_pattern, new_key):
                prefix = re.findall(ln1_pattern, new_key)[0]
                new_key = new_key.replace(prefix, f"{prefix[:-5]}.attn.ln")
            elif re.findall(ln2_pattern, new_key):
                prefix = re.findall(ln2_pattern, new_key)[0]
                new_key = new_key.replace(prefix, f"{prefix[:-5]}.mlp.ln")
            state_dict[new_key] = val.clone()

        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
        )
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks"))
        )

        self.clip = CLIP(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
            adapter,
        ).float()

        missing_keys, unexpected_keys, error_msgs = [], [], []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict=True,
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
                error_msgs=error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self, prefix="")
        convert_weights(self.clip)

    # ------------------------------------------------------------------
    # 工具 (保持不变)
    # ------------------------------------------------------------------
    def to_device(self, tensor):
        if tensor is None:
            return None
        if isinstance(tensor, list):
            try:
                tensor = torch.stack(tensor)
            except Exception:
                tensor = torch.tensor(tensor)
        return tensor.to(self.clip.logit_scale.device)

    def _to_int_list(self, x):
        out = []

        def rec(v):
            if torch.is_tensor(v):
                for t in v.view(-1).tolist():
                    rec(t)
            elif isinstance(v, (list, tuple)):
                for u in v:
                    rec(u)
            else:
                out.append(int(v))

        rec(x)
        seen = set()
        uniq = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def process_caption(self, caption_batch: List[str]):
        max_words = self.max_words
        tokens = [
            [SPECIAL_TOKENS["CLS_TOKEN"]] + self.tokenizer.tokenize(c)[: max_words - 2] + [SPECIAL_TOKENS["SEP_TOKEN"]]
            for c in caption_batch
        ]
        token_ids = [
            self.tokenizer.convert_tokens_to_ids(t) + [0] * (max_words - len(t))
            for t in tokens
        ]
        caption_tensors = self.to_device(torch.tensor(token_ids, dtype=torch.long))
        caption_masks = torch.ones_like(caption_tensors)
        caption_masks[caption_tensors == 0] = 0
        return caption_tensors, caption_masks

    def _prompts_for_class(self, class_idx: int) -> List[str]:
        assert self.sub_act_map is not None and len(self.all_labels) > 0, \
            "没有 sub_act_map，无法构建文本原型"
        if class_idx < 0 or class_idx >= len(self.all_labels):
            raise IndexError(f"[BaselineModel] class_idx 越界: {class_idx}")

        label_key = self.all_labels[class_idx]
        entry = self.sub_act_map[label_key]

        if isinstance(entry, dict):
            sub_acts = entry.get("sub_act_en_li", [])
            label_text = entry.get("label", label_key)
        else:
            sub_acts = list(entry)
            label_text = label_key

        if not sub_acts:
            sub_acts = [label_text]

        prompts = [f"A video of action about {label_text}: {sa}" for sa in sub_acts]
        return prompts

    def _build_episode_text_proto(self, class_ids: torch.Tensor) -> torch.Tensor:
        device = self.clip.logit_scale.device
        target_dtype = torch.float32 if self.fp32 else next(self.clip.parameters()).dtype

        protos = []
        for cid in class_ids.tolist():
            prompts = self._prompts_for_class(int(cid))
            tokens, masks = self.process_caption(prompts)
            with autocast(device_type=device.type, dtype=target_dtype):
                tfeat = self.clip.encode_text(tokens, mask=masks)
            tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-6)
            proto = tfeat.mean(dim=0)
            protos.append(proto)

        ep_text_proto = torch.stack(protos, dim=0).to(device)
        ep_text_proto = ep_text_proto / (ep_text_proto.norm(dim=-1, keepdim=True) + 1e-6)
        return ep_text_proto

    def _flatten_video_episode(
            self,
            videos: Union[torch.Tensor, List[torch.Tensor]],
            v_mask: Union[torch.Tensor, List[torch.Tensor]],
    ):
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0)
        if isinstance(v_mask, list):
            v_mask = torch.stack(v_mask, dim=0)

        if videos.dim() == 7:
            E, W_, NQ, T, C, H, W = videos.shape
            videos = videos.view(E * W_ * NQ, T, C, H, W)
            v_mask = v_mask.view(E * W_ * NQ, T)
        elif videos.dim() == 6:
            W_, NQ, T, C, H, W = videos.shape
            videos = videos.view(W_ * NQ, T, C, H, W)
            v_mask = v_mask.view(W_ * NQ, T)
        elif videos.dim() == 5:
            pass
        else:
            raise RuntimeError(f"[BaselineModel] 不支持的视频维度: {videos.shape}")

        return videos, v_mask

    def encode_video_batch(self, videos: torch.Tensor, v_mask: torch.Tensor):
        device = self.clip.logit_scale.device
        videos = videos.to(device)
        v_mask = v_mask.to(device)

        B, T, C, H, W = videos.shape
        videos_flat = videos.view(B * T, C, H, W)
        v_mask_flat = v_mask.view(B * T)

        chunk = self.vis_chunk_size
        visual_tokens: List[torch.Tensor] = []
        for start in range(0, B * T, chunk):
            end = min(start + chunk, B * T)
            images_chunk = videos_flat[start:end]
            mask_chunk = v_mask_flat[start:end].unsqueeze(1)

            with autocast(device_type=device.type, dtype=self.training_type):
                vt = self.clip.encode_image(images_chunk, mask=mask_chunk)
            vt = vt.float()
            visual_tokens.append(vt)

        visual_tokens = torch.cat(visual_tokens, dim=0)
        visual_tokens = visual_tokens.view(B, T, -1)

        v_mask_un = v_mask.to(dtype=torch.float32).unsqueeze(-1)
        visual_tokens = visual_tokens * v_mask_un
        denom = v_mask_un.sum(dim=1)
        denom[denom == 0] = 1.0
        video_emb = visual_tokens.sum(dim=1) / denom
        video_emb = video_emb / (video_emb.norm(dim=-1, keepdim=True) + 1e-6)
        return video_emb

    # ------------------------------------------------------------------
    # 训练：episodic few-shot loss (AMAL 修改版)
    # ------------------------------------------------------------------
    def forward_episode_train(self, batch: Dict[str, Any]):
        device = self.clip.logit_scale.device

        # 1. 准备 Support Data
        s_videos = batch["support_videos"]
        s_v_mask = batch["support_v_mask"]
        s_videos, s_v_mask = self._flatten_video_episode(s_videos, s_v_mask)

        # =============================================================
        # AMAL Step 1: Inference (生成参数)
        # =============================================================
        self.reset_task_parameters()
        with torch.no_grad():
            s_feat_raw = self.encode_video_batch(s_videos, s_v_mask)

        # 生成 Delta (Sigma)
        task_sigma = self.task_encoder(s_feat_raw)

        # =============================================================
        # AMAL Step 2: Meta-Adaptation (注入参数)
        # =============================================================
        self.inject_task_parameters(task_sigma)

        # =============================================================
        # AMAL Step 3: Outer Loop (推理 & Loss)
        # =============================================================
        # 3.1 重新编码 Support (特化后的特征)
        s_feat = self.encode_video_batch(s_videos, s_v_mask)
        s_label = torch.as_tensor(batch["support_label"], device=device).long().view(-1)

        # 3.2 编码 Query (特化后的特征)
        q_videos = batch["video_batch"]
        q_v_mask = batch["v_mask"]
        q_videos, q_v_mask = self._flatten_video_episode(q_videos, q_v_mask)
        q_feat = self.encode_video_batch(q_videos, q_v_mask)

        # --- 原型网络逻辑 (不变) ---
        if "episode_class_ids" in batch and batch["episode_class_ids"] is not None:
            ep_ids = self._to_int_list(batch["episode_class_ids"])
        else:
            ep_ids = self._to_int_list(torch.unique(s_label).long())
        ep_cls = torch.tensor(ep_ids, device=device, dtype=torch.long)
        ep_cls_list = ep_ids

        ep_text_proto = self._build_episode_text_proto(ep_cls)

        alpha = self.support_proto_alpha
        cls2feats = defaultdict(list)
        for feat, lab in zip(s_feat, s_label):
            cls2feats[int(lab.item())].append(feat)

        fused_proto = []
        for idx, cid in enumerate(ep_cls_list):
            tproto = ep_text_proto[idx]
            if cid in cls2feats and len(cls2feats[cid]) > 0:
                vfeats = torch.stack(cls2feats[cid], dim=0)
                vfeats = vfeats / (vfeats.norm(dim=-1, keepdim=True) + 1e-6)
                vproto = vfeats.mean(dim=0)
                newp = (1 - alpha) * tproto + alpha * vproto if alpha > 0 else tproto
                newp = newp / (newp.norm(dim=-1, keepdim=True) + 1e-6)
                fused_proto.append(newp)
            else:
                fused_proto.append(tproto)
        ep_proto = torch.stack(fused_proto, dim=0)

        logit_scale = self.clip.logit_scale.exp().to(device)
        logits = logit_scale * torch.matmul(q_feat, ep_proto.t())

        q_label = torch.as_tensor(batch["label"], device=device).long().view(-1)
        gid2lid = {int(g): i for i, g in enumerate(ep_cls_list)}
        q_local = torch.tensor([gid2lid[int(g)] for g in q_label.tolist()],
                               device=device).long()

        loss = F.cross_entropy(logits, q_local)

        # 4. 清理
        self.reset_task_parameters()

        return loss

    # ------------------------------------------------------------------
    # 推理：few-shot (AMAL 修改版)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_fewshot_infer(self, batch: Dict[str, Any]):
        device = self.clip.logit_scale.device

        # 1. Support Data
        s_videos = batch["support_videos"]
        s_v_mask = batch["support_v_mask"]
        s_labels = batch["support_label"]
        s_videos, s_v_mask = self._flatten_video_episode(s_videos, s_v_mask)
        s_labels = torch.as_tensor(s_labels, device=device).long().view(-1)

        # 2. AMAL Generate & Inject
        self.reset_task_parameters()
        s_feat_raw = self.encode_video_batch(s_videos, s_v_mask)
        task_sigma = self.task_encoder(s_feat_raw)
        self.inject_task_parameters(task_sigma)

        # 3. Re-Encode Support & Query
        s_feat = self.encode_video_batch(s_videos, s_v_mask)
        q_videos = batch["video_batch"]
        q_v_mask = batch["v_mask"]
        q_videos, q_v_mask = self._flatten_video_episode(q_videos, q_v_mask)
        q_feat = self.encode_video_batch(q_videos, q_v_mask)

        # --- 原型网络逻辑 (不变) ---
        if "episode_class_ids" in batch and batch["episode_class_ids"] is not None:
            ep_ids = self._to_int_list(batch["episode_class_ids"])
        else:
            ep_ids = self._to_int_list(torch.unique(s_labels).long())
        ep_cls = torch.tensor(ep_ids, device=device, dtype=torch.long)
        ep_cls_list = ep_ids

        ep_text_proto = self._build_episode_text_proto(ep_cls)

        alpha = self.support_proto_alpha
        if alpha > 0:
            cls2feats = defaultdict(list)
            for feat, lab in zip(s_feat, s_labels):
                cls2feats[int(lab.item())].append(feat)
            ep_visual_proto = []
            for idx, cid in enumerate(ep_cls_list):
                if cid in cls2feats:
                    feats = torch.stack(cls2feats[cid], dim=0)
                    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
                    vproto = feats.mean(dim=0)
                    tproto = ep_text_proto[idx]
                    newp = (1 - alpha) * tproto + alpha * vproto
                    newp = newp / (newp.norm(dim=-1, keepdim=True) + 1e-6)
                    ep_visual_proto.append(newp)
                else:
                    ep_visual_proto.append(ep_text_proto[idx])
            ep_proto = torch.stack(ep_visual_proto, dim=0)
        else:
            ep_proto = ep_text_proto

        logit_scale = self.clip.logit_scale.exp().to(device)
        sim_scores = logit_scale * torch.matmul(q_feat, ep_proto.t())
        pred_labels_local = torch.argmax(sim_scores, dim=1)

        pred_global = torch.tensor(
            [ep_cls_list[int(i)] for i in pred_labels_local.tolist()],
            device=sim_scores.device
        )

        self.reset_task_parameters()
        return pred_global, sim_scores, ep_cls

    def forward(self, batch: Dict[str, Any]):
        if self.training:
            return self.forward_episode_train(batch)
        else:
            if ("support_videos" in batch) and ("support_v_mask" in batch) and ("support_label" in batch):
                return self.forward_fewshot_infer(batch)
            raise RuntimeError("推理模式仅支持 few-shot，请提供 support 数据。")

