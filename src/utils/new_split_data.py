import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ===================== 配置区 =====================
dataset_root = ROOT / "data/NEU-DET"
images_src = dataset_root / "images/train"
labels_src = dataset_root / "labels/train"

seed_ratio = 0.2        # 少量有标注
unlabeled_ratio = 0.8   # 模拟无标注

seed = 42
# =================================================

assert abs(seed_ratio + unlabeled_ratio - 1.0) < 1e-6

random.seed(seed)

# ---------- 创建目录 ----------
dirs = [
    dataset_root / "seed/images/train",
    dataset_root / "seed/labels/train",
    dataset_root / "unlabeled/images/train",
    dataset_root / "unlabeled/labels_hidden/train",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# ---------- 读取所有图片 ----------
image_files = list(images_src.glob("*.jpg")) + list(images_src.glob("*.png"))
image_files.sort()
random.shuffle(image_files)

num_total = len(image_files)
num_seed = int(num_total * seed_ratio)

seed_files = image_files[:num_seed]
unlabeled_files = image_files[num_seed:]

# ---------- 拷贝 Seed（有标注） ----------
for img_path in seed_files:
    label_path = labels_src / f"{img_path.stem}.txt"
    if not label_path.exists():
        print(f"[WARNING] 标签不存在: {label_path.name}")
        continue

    shutil.copy(img_path, dataset_root / "seed/images/train" / img_path.name)
    shutil.copy(label_path, dataset_root / "seed/labels/train" / label_path.name)

# ---------- 拷贝 Unlabeled（隐藏标签） ----------
for img_path in unlabeled_files:
    label_path = labels_src / f"{img_path.stem}.txt"
    if not label_path.exists():
        print(f"[WARNING] 标签不存在: {label_path.name}")
        continue

    shutil.copy(img_path, dataset_root / "unlabeled/images/train" / img_path.name)

    # 标签隐藏（仅保存用于评估，不参与训练）
    shutil.copy(label_path, dataset_root / "unlabeled/labels_hidden/train" / label_path.name)

# ---------- 记录划分信息 ----------
with open(dataset_root / "split_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Total images: {num_total}\n")
    f.write(f"Seed (labeled): {len(seed_files)}\n")
    f.write(f"Unlabeled (simulated): {len(unlabeled_files)}\n")

print("===== 半监督数据集构建完成 =====")
print(f"Total      : {num_total}")
print(f"Seed       : {len(seed_files)}")
print(f"Unlabeled  : {len(unlabeled_files)}")
