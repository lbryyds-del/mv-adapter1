#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把动作分类数据集的类别列表，按 base / val / novel 三份划分并保存成 txt。

示例：
python tools/split_fewshot_classes.py \
    --root /home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/train-frames \
    --dataset hmdb51 \
    --out-dir /home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51_splits
"""
import os
import argparse
import random

HMDB51_SPLIT = (31, 10, 10)
UCF101_SPLIT = (70, 10, 21)


def write_txt(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(x + "\n")
    print(f"[save] {path} ({len(items)} classes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="类别所在的帧目录，比如 .../data/hmdb51/train-frames")
    parser.add_argument("--dataset", choices=["hmdb51", "ucf101", "custom"],
                        default="hmdb51",
                        help="选择默认的划分数量")
    parser.add_argument("--out-dir", required=True,
                        help="输出 txt 的目录")
    parser.add_argument("--base", type=int, default=None,
                        help="custom 模式下：base 类别数")
    parser.add_argument("--val", type=int, default=None,
                        help="custom 模式下：val 类别数")
    parser.add_argument("--novel", type=int, default=None,
                        help="custom 模式下：novel 类别数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    # 1) 列类别
    all_classes = [d for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d))]
    all_classes = sorted(all_classes)
    num_classes = len(all_classes)
    print(f"[info] found {num_classes} classes in {args.root}")

    # 2) 定数量
    if args.dataset == "hmdb51":
        base_n, val_n, novel_n = HMDB51_SPLIT
    elif args.dataset == "ucf101":
        base_n, val_n, novel_n = UCF101_SPLIT
    else:
        assert args.base and args.val and args.novel, \
            "custom 模式下必须给 --base --val --novel"
        base_n, val_n, novel_n = args.base, args.val, args.novel

    assert base_n + val_n + novel_n <= num_classes, \
        f"要的({base_n}+{val_n}+{novel_n})>实际({num_classes})，调小点"

    # 3) 打乱切片
    random.seed(args.seed)
    random.shuffle(all_classes)

    base_classes = all_classes[:base_n]
    val_classes = all_classes[base_n:base_n + val_n]
    novel_classes = all_classes[base_n + val_n:base_n + val_n + novel_n]

    # 4) 保存
    base_txt = os.path.join(args.out_dir, "base_classes.txt")
    val_txt = os.path.join(args.out_dir, "val_classes.txt")
    novel_txt = os.path.join(args.out_dir, "novel_classes.txt")

    write_txt(base_txt, base_classes)
    write_txt(val_txt, val_classes)
    write_txt(novel_txt, novel_classes)

    print("[done]")


if __name__ == "__main__":
    main()
