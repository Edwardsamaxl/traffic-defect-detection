import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ===================== 配置区 =====================
dataset_root = ROOT / "data/NEU-DET"

images_src = dataset_root / "images_all"
labels_src = dataset_root / "labels_all"

images_dst = dataset_root / "images"
labels_dst = dataset_root / "labels"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

seed = 42
# =================================================

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

random.seed(seed)

# 创建目标目录
for split in ["train", "val", "test"]:
    (images_dst / split).mkdir(parents=True, exist_ok=True)
    (labels_dst / split).mkdir(parents=True, exist_ok=True)

# 读取所有图片
image_files = list(images_src.glob("*.jpg")) + list(images_src.glob("*.png"))
image_files.sort()
random.shuffle(image_files)

num_total = len(image_files)
num_train = int(num_total * train_ratio)
num_val = int(num_total * val_ratio)

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

def copy_split(files, split):
    for img_path in files:
        label_path = labels_src / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARNING] 标签不存在: {label_path.name}")
            continue

        shutil.copy(img_path, images_dst / split / img_path.name)
        shutil.copy(label_path, labels_dst / split / label_path.name)

copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")

print("===== 数据集划分完成 =====")
print(f"Total : {num_total}")
print(f"Train : {len(train_files)}")
print(f"Val   : {len(val_files)}")
print(f"Test  : {len(test_files)}")
