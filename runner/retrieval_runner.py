# runner/retrieval_runner.py
import math
import os.path as osp
from time import time
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from tqdm import tqdm

from .base_runner import Runner
from registry import RUNNERS
from util import my_log, AverageMeter, my_all_gather


def sec2timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _split_batch_into_episodes(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    dataloader 验证是 batch_size>1 的话：
    - list(episode0, episode1, ...) 需要拆回单条 episode，方便后面按 1/5-shot 评
    """
    # 推断 episode 数量
    if "support_videos" in batch and isinstance(batch["support_videos"], list):
        num_ep = len(batch["support_videos"])
    elif "video_batch" in batch and isinstance(batch["video_batch"], list):
        num_ep = len(batch["video_batch"])
    else:
        return [batch]  # 已是单条

    episodes: List[Dict[str, Any]] = []
    for i in range(num_ep):
        ep: Dict[str, Any] = {}
        # list_plus 类字段
        for k in ["video_batch", "support_videos"]:
            if k in batch and isinstance(batch[k], list):
                ep[k] = batch[k][i]
        # tensor 类字段（第 0 维是 batch）
        for k in ["v_mask", "support_v_mask", "label", "support_label", "episode_class_ids"]:
            if k in batch and torch.is_tensor(batch[k]):
                ep[k] = batch[k][i]
        episodes.append(ep)
    return episodes

def _slice_support(batch: Dict[str, Any], shot: int) -> Dict[str, Any]:
    """
    把单条 episode 的 support 切到指定 shot（改为随机采样）
    """
    new_batch = dict(batch)

    s_vid = batch["support_videos"]          # [W*S, T, 3, H, W]
    s_mask = batch["support_v_mask"]         # [W*S, T]
    s_lab = batch["support_label"]           # [W*S]

    # way
    if "episode_class_ids" in batch and batch["episode_class_ids"] is not None:
        ep_ids = batch["episode_class_ids"]
        way = int(ep_ids.numel()) if torch.is_tensor(ep_ids) else len(ep_ids)
    else:
        way = int(torch.unique(s_lab).numel())

    total_sup = s_vid.shape[0]
    max_shot = total_sup // way
    use_shot = min(shot, max_shot)

    # reshape → [way, max_shot, ...]
    s_vid = s_vid.view(way, max_shot, *s_vid.shape[1:])
    s_mask = s_mask.view(way, max_shot, *s_mask.shape[1:])
    s_lab = s_lab.view(way, max_shot)

    # 在每个类内部随机拿 use_shot 个 support
    # 这里对所有类共用一套下标，保证广播一致
    if use_shot < max_shot:
        # device 和 dtype 跟 s_vid 一致
        perm = torch.randperm(max_shot, device=s_vid.device)[:use_shot]
        s_vid = s_vid[:, perm]
        s_mask = s_mask[:, perm]
        s_lab = s_lab[:, perm]
    else:
        # shot >= max_shot，等价于用全部 support
        pass

    s_vid = s_vid.contiguous().view(way * use_shot, *s_vid.shape[2:])
    s_mask = s_mask.contiguous().view(way * use_shot, *s_mask.shape[2:])
    s_lab = s_lab.contiguous().view(way * use_shot)

    new_batch["support_videos"] = s_vid
    new_batch["support_v_mask"] = s_mask
    new_batch["support_label"] = s_lab
    return new_batch




@RUNNERS.register_module()
class RetrievalRunner(Runner):
    """
    few-shot episodic 的 runner（评测方式对齐 D2ST / Task-Adapter）：
    - 单 episode 评
    - 支持多种 shot（从 cfg.test_dataset.eval_shots 读取）
    """

    def __init__(self, cfg, model, train_dataloader, val_dataloader, optimizer,
                 train_sampler=None, val_sampler=None) -> None:
        super().__init__(cfg, model, train_dataloader, val_dataloader, optimizer)
        self.results: List[Dict[str, float]] = []
        self.loss_meter = AverageMeter()
        self.time_meter = AverageMeter()
        self.f_meter = AverageMeter()
        self.b_meter = AverageMeter()
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        mdl = self.model.module if hasattr(self.model, "module") else self.model
        self.all_labels = getattr(mdl, "all_labels", [])
        self.label2id = {name: i for i, name in enumerate(self.all_labels)}

        _rank = dist.get_rank() if dist.is_initialized() else 0
        if _rank == 0:
            print(f"[EVAL] episodic few-shot mode, model labels={len(self.all_labels)}")

    # ========================= train =========================
    def train(self):
        model = self.model
        model.train()
        torch.cuda.empty_cache()

        log_step = self.cfg.log_step
        start_time = time()
        rank = dist.get_rank() if dist.is_initialized() else 0

        train_iter = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.max_epochs}",
            disable=(rank != 0)
        )

        for step, batch in enumerate(train_iter):
            t1 = time()
            loss = model(batch)
            self.f_meter.update(time() - t1)
            self.loss_meter.update(loss.item())

            # backward
            t1 = time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.b_meter.update(time() - t1)

            # 限制 CLIP 的 logit_scale
            max_tau = 100 if getattr(self.cfg, "tau", "constant") != "sqrt" else \
                100 - 80 * math.sqrt(step / max(1, self.max_steps))
            target = model.module if hasattr(model, "module") else model
            try:
                if hasattr(target, "clip") and hasattr(target.clip, "logit_scale") and target.clip.logit_scale is not None:
                    torch.clamp_(target.clip.logit_scale.data, max=math.log(max_tau))
            except Exception:
                pass

            self.global_step += 1

            if rank == 0 and (step + 1) % log_step == 0:
                lr_list = sorted(list(set(self.optimizer.get_lr())))
                lr_str = ",".join([f"{lr:.2e}" for lr in lr_list])
                train_iter.set_postfix(loss=f"{self.loss_meter.avg:.4f}", lr=lr_str)

                msg = (
                    f"[Train] epoch={self.epoch + 1}/{self.max_epochs} "
                    f"step={step + 1}/{self.max_steps} "
                    f"loss(cur/avg)={self.loss_meter.val:.4f}/{self.loss_meter.avg:.4f} "
                    f"fwd_time(avg)={self.f_meter.avg:.4f}s bwd_time(avg)={self.b_meter.avg:.4f}s "
                    f"lr={lr_str}"
                )
                my_log(msg, logger="result")

            self.time_meter.update(time() - start_time)
            start_time = time()

    # ========================= evaluate =========================
    @torch.no_grad()
    def evaluate(self):
        if self.val_dataloader is None:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                my_log("[RetrievalRunner] val_dataloader is None, skip evaluation.", logger="result")
            return None

        model = self.model
        model.eval()

        # 想评多少条、评哪些 shot
        td_cfg = getattr(self.cfg, "test_dataset", {})
        max_eval_eps = td_cfg.get("eval_episodes", 10000)
        eval_shots = td_cfg.get("eval_shots", [1, 5])
        eval_log_interval = td_cfg.get("log_interval", 20)

        rank = dist.get_rank() if dist.is_initialized() else 0
        val_iter = tqdm(
            self.val_dataloader,
            desc="Validating (few-shot episodic)",
            disable=(rank != 0)
        )

        shot2acc: Dict[int, List[float]] = {s: [] for s in eval_shots}

        seen_eps = 0
        for batch in val_iter:
            # 拆成多条 episode
            ep_list = _split_batch_into_episodes(batch)

            for ep in ep_list:
                for shot in eval_shots:
                    sbatch = _slice_support(ep, shot)
                    out = model(sbatch)
                    if not (isinstance(out, (list, tuple)) and len(out) >= 2):
                        raise RuntimeError("model should return (pred_global, sim_scores, ep_cls) in eval")
                    pred_global, sim_scores, ep_cls = out

                    labels = sbatch["label"]
                    if torch.is_tensor(labels):
                        labels = labels.view(-1).cpu()
                    else:
                        labels = torch.tensor(labels, dtype=torch.long)
                    pred_global = pred_global.detach().cpu().view(-1)

                    acc = (pred_global == labels).float().mean().item() * 100.0
                    shot2acc[shot].append(acc)

                if rank == 0 and eval_log_interval > 0 and (seen_eps + 1) % eval_log_interval == 0:
                    detail_msg = [
                        f"shot={s}: {torch.tensor(a).float().mean().item():.2f}% ({len(a)} eps)"
                        for s, a in shot2acc.items() if len(a) > 0
                    ]
                    msg = (
                        f"[Eval Progress] processed {seen_eps + 1} episodes "
                        f"| " + " | ".join(detail_msg)
                    )
                    my_log(msg, logger="result")

                seen_eps += 1
                if seen_eps >= max_eval_eps:
                    break

            if seen_eps >= max_eval_eps:
                break

        # ====== 多卡汇总并打印 ======
        results = {}
        for shot, acc_list in shot2acc.items():
            device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
            acc_tensor = torch.tensor(acc_list, dtype=torch.float32, device=device)
            gathered = my_all_gather(acc_tensor)
            acc_all = torch.cat([t.cpu() for t in gathered], dim=0)

            if rank != 0:
                continue

            n_ep = acc_all.numel()
            mean_acc = float(acc_all.mean().item()) if n_ep > 0 else 0.0
            std_acc = float(acc_all.std(unbiased=True).item()) if n_ep > 1 else 0.0
            ci95 = 1.96 * std_acc / (n_ep ** 0.5) if n_ep > 1 else 0.0

            msg = f"[FewShot Eval] shot={shot} | Episodes={n_ep} | Top1: {mean_acc:.2f}±{ci95:.2f}"
            my_log(msg, logger="result")

            results[f"shot{shot}_top1_mean"] = mean_acc
            results[f"shot{shot}_top1_ci95"] = ci95

        if rank == 0:
            return results
        else:
            return None

    # ========================= run =========================
    def run(self):
        cfg = self.cfg.train_cfg

        # -------- 测试专用模式：只评一次就退出 --------
        if cfg.get("max_epochs", 1) <= 0:
            _rank = dist.get_rank() if dist.is_initialized() else 0
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(0)
            elif (self.val_dataloader is not None and
                  hasattr(self.val_dataloader, "sampler") and
                  isinstance(self.val_dataloader.sampler, torch.utils.data.distributed.DistributedSampler)):
                self.val_dataloader.sampler.set_epoch(0)

            self.evaluate()
            if dist.is_initialized():
                dist.barrier()
            return

        # -------- 训练 + 周期性评估路径 --------
        # 可选：开头先评一次
        if cfg.get("val_begin", 0) == 0 and self.val_dataloader is not None:
            self.evaluate()

        start = time()

        for e in range(cfg.get("max_epochs", 1)):
            self.epoch = e

            # sampler epoch
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(e)
            elif hasattr(self.train_dataloader, "sampler") and isinstance(
                self.train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(e)

            if self.val_sampler is not None:
                self.val_sampler.set_epoch(0)
            elif (self.val_dataloader is not None and
                  hasattr(self.val_dataloader, "sampler") and
                  isinstance(self.val_dataloader.sampler, torch.utils.data.distributed.DistributedSampler)):
                self.val_dataloader.sampler.set_epoch(0)

            self.model.epoch = e

            self.train()
            res = self.evaluate()

            if dist.is_initialized() and dist.get_rank() > 0:
                continue

            if res is not None:
                # 用 shot=1 的结果做保存依据
                top1_acc = res.get("shot1_top1_mean", 0.0)
                res["model"] = self.save_model(top1_acc, 0.0)
                self.results.append(res)

            my_log(
                f"Epoch {e + 1}/{self.max_epochs} finished in {sec2timestamp(time() - start)}",
                logger="result"
            )
            start = time()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if len(self.results) > 0:
                self.log_best()

    def log_best(self):
        best = None
        for r in self.results:
            v = r.get("shot1_top1_mean", 0.0)
            if best is None or v > best.get("shot1_top1_mean", 0.0):
                best = r
        if best is not None:
            my_log(f"\n\nBest Result (shot=1):\n{best}\n", logger="result")

    def save_model(self, top1_acc=0.0, top5_acc=0.0):
        model = self.model
        model_to_save = model.module if hasattr(model, "module") else model
        postfix = f"{self.epoch + 1}e_top1{int(round(top1_acc, 2) * 100):04d}.pth"
        output_model_file = osp.join(self.work_dir, postfix)
        trainable_state_dict = {
            k: v for k, v in model_to_save.named_parameters() if v.requires_grad
        }
        torch.save(trainable_state_dict, output_model_file)
        my_log(f"Model saved to {output_model_file}", logger="result")
        return output_model_file
