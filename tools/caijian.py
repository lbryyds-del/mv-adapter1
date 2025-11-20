import os

in_txt = "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/val_few_shot.txt"      # 你的原始txt（现在是路径）
out_txt = "/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/hmdb51_classes_val.txt"  # 想要的“类名单”

cls_set = set()
with open(in_txt, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 标准化分隔符
        line = line.replace("\\", "/")
        parts = line.split("/")
        if len(parts) < 2:
            continue
        cls_name = parts[-2]      # 倒数第二层目录，就是类名
        cls_set.add(cls_name)

with open(out_txt, "w", encoding="utf-8") as f:
    for c in sorted(cls_set):
        f.write(c + "\n")

print("written:", out_txt, "num classes:", len(cls_set))
