import os
import os.path as osp
import shutil
import argparse

def read_list(p):
    with open(p, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]

def safe_mkdir(d):
    os.makedirs(d, exist_ok=True)

def copy_or_link_dir(src, dst, mode='link'):
    """
    把 src 这个目录整个放到 dst 下面：
    - mode='link'：创建软链接，快、不占空间（推荐）
    - mode='copy'：真的拷贝一份，占空间大但最保险
    """
    if osp.exists(dst):
        return
    if mode == 'copy':
        shutil.copytree(src, dst)
    else:
        # 软链接目录
        os.symlink(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-root',
                    default='/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/HMDB51',
                    help='原始所有类的frames所在目录')
    ap.add_argument('--train-list',
                    default='/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/hmdb51_splits/base_classes.txt')
    ap.add_argument('--val-list',
                    default='/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/hmdb51_splits/val_classes.txt')
    ap.add_argument('--test-list',
                    default='/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51/hmdb51_splits/novel_classes.txt')
    ap.add_argument('--dst-root',
                    default='/home/liborui/PycharmProjects/MV-Adapter-main/data/hmdb51',
                    help='要在这里面生成 train-frames / val-frames / test-frames')
    ap.add_argument('--mode', choices=['link', 'copy'], default='link',
                    help='link: 建软链接; copy: 真拷贝')
    args = ap.parse_args()

    src_root = args.src_root
    train_classes = set(read_list(args.train_list))
    val_classes   = set(read_list(args.val_list))
    test_classes  = set(read_list(args.test_list))

    # 目标目录
    train_dst_root = osp.join(args.dst_root, 'train-frames')
    val_dst_root   = osp.join(args.dst_root, 'val-frames')
    test_dst_root  = osp.join(args.dst_root, 'test-frames')
    safe_mkdir(train_dst_root)
    safe_mkdir(val_dst_root)
    safe_mkdir(test_dst_root)

    # 扫描源目录
    for cls in sorted(os.listdir(src_root)):
        cls_dir = osp.join(src_root, cls)
        if not osp.isdir(cls_dir):
            continue

        # 看这个类属于哪一份
        target_root = None
        if cls in train_classes:
            target_root = train_dst_root
        elif cls in val_classes:
            target_root = val_dst_root
        elif cls in test_classes:
            target_root = test_dst_root
        else:
            # 不在三个列表里的忽略
            print(f'[skip] class "{cls}" not in any list.')
            continue

        # 在目标里建这个类的目录
        dst_cls_dir = osp.join(target_root, cls)
        safe_mkdir(dst_cls_dir)

        # 这个类下面是一堆视频目录，全搬过去
        for vid in sorted(os.listdir(cls_dir)):
            src_vid_dir = osp.join(cls_dir, vid)
            if not osp.isdir(src_vid_dir):
                continue
            dst_vid_dir = osp.join(dst_cls_dir, vid)
            copy_or_link_dir(src_vid_dir, dst_vid_dir, mode=args.mode)
        print(f'[done] {cls} -> {target_root}')

    print('All done.')

if __name__ == '__main__':
    main()
