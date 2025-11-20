import os

splits_dir = ""  # 解压后目录名字，按你实际的改
split_id = 1  # 1 / 2 / 3

classes = sorted([
    d[:-len(f"_test_split{split_id}.txt")]
    for d in os.listdir(splits_dir)
    if d.endswith(f"_test_split{split_id}.txt")
])

train_out = open(f"hmdb51_train_split{split_id}.txt", "w")
test_out = open(f"hmdb51_test_split{split_id}.txt", "w")

for cls_id, cls_name in enumerate(classes):
    train_file = os.path.join(splits_dir, f"{cls_name}_train_split{split_id}.txt")
    test_file = os.path.join(splits_dir, f"{cls_name}_test_split{split_id}.txt")

    # train 文件里：1=用作训练，0=不用
    with open(train_file, "r") as f:
        for line in f:
            video, flag = line.strip().split()
            flag = int(flag)
            if flag == 1:
                # 写：视频名 类别id
                train_out.write(f"{video} {cls_id}\n")

    # test 文件里：2=用作测试，0=不用
    with open(test_file, "r") as f:
        for line in f:
            video, flag = line.strip().split()
            flag = int(flag)
            if flag == 2:
                test_out.write(f"{video} {cls_id}\n")

train_out.close()
test_out.close()
print("done.")
