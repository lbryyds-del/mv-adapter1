import os, json, re
from os import path as osp
from collections import OrderedDict

UF_ROOT = "/home/liborui/PycharmProjects/MV-Adapter-main"
SUB_ACT_JSON = f"{UF_ROOT}/data/action_sub_acts.json"
SCAN_DIRS = [
    f"{UF_ROOT}/data/ufc101/train-frames",
    f"{UF_ROOT}/data/ufc101/test-frames",
]

def camel_to_space(name: str) -> str:
    s = re.sub(r'(?<!^)([A-Z])', r' \1', name).strip()
    return s.lower()

def load_classes_from_dirs(dirs):
    classes = set()
    for d in dirs:
        if not osp.isdir(d):
            continue
        for c in os.listdir(d):
            cdir = osp.join(d, c)
            if osp.isdir(cdir):
                classes.add(c)
    return sorted(classes)

def main():
    assert osp.isfile(SUB_ACT_JSON), f"缺少 {SUB_ACT_JSON}"
    with open(SUB_ACT_JSON,"r") as f:
        subacts = json.load(f)

    # 统一成 {ClassName: {"label": <prompt_label>, "sub_act_en_li":[...]} } 结构
    norm = OrderedDict()
    for k,v in subacts.items():
        if isinstance(v, dict) and "sub_act_en_li" in v:
            # 键就是类名
            label_text = v.get("label", camel_to_space(k))
            norm[k] = {"label": label_text, "sub_act_en_li": v["sub_act_en_li"]}
        else:
            # v 可能就是 list（你的简写），k 是类名或自然语
            # 这里假设 k 是类名，若不是，之后用目录名覆盖
            li = v if isinstance(v,list) else v.get("sub_act_en_li",[])
            norm[k] = {"label": camel_to_space(k), "sub_act_en_li": li}

    classes = load_classes_from_dirs(SCAN_DIRS)
    added, missing = 0, []
    for cname in classes:
        if cname not in norm:
            added += 1
            norm[cname] = {
                "label": camel_to_space(cname),
                "sub_act_en_li": [
                    "beginning stage (to be refined).",
                    "process stage (to be refined).",
                    "ending stage (to be refined).",
                ]
            }

    # 只保留目录中存在的类（可选）
    # norm = OrderedDict((k, norm[k]) for k in classes if k in norm)

    with open(SUB_ACT_JSON, "w") as f:
        json.dump(norm, f, indent=2, ensure_ascii=False)
    print(f"共 {len(classes)} 个目录类；为缺失类补齐 {added} 条。已回写 {SUB_ACT_JSON}")

if __name__ == "__main__":
    main()
