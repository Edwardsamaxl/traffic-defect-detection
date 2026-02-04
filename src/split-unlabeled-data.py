import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ===================== 配置区 =====================
dataset_root = ROOT / "data/NEU-DET-semi"

images_src = ROOT / "data/NEU-DET-semi/images_all"
labels_src = ROOT / "data/NEU-DET-semi/labels_all"

seed_ratio = 0.2        # 少量有标注
unlabeled_ratio = 0.6   # 模拟无标注
test_ratio = 0.2        # 独立测试

seed_train_ratio = 0.8  # seed 内部 train/val

seed = 42
# =================================================

assert abs(seed_ratio + unlabeled_ratio + test_ratio - 1.0) < 1e-6

random.seed(seed)

# ---------- 创建目录 ----------
dirs = [
    dataset_root / "seed/images/train",
    dataset_root / "seed/images/val",
    dataset_root / "seed/labels/train",
    dataset_root / "seed/labels/val",

    dataset_root / "unlabeled/images/train",
    dataset_root / "unlabeled/labels_hidden/train",

    dataset_root / "test/images",
    dataset_root / "test/labels",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# ---------- 读取所有图片 ----------
image_files = list(images_src.glob("*.jpg")) + list(images_src.glob("*.png"))
image_files.sort()
random.shuffle(image_files)

num_total = len(image_files)
num_seed = int(num_total * seed_ratio)
num_unlabeled = int(num_total * unlabeled_ratio)

seed_files = image_files[:num_seed]
unlabeled_files = image_files[num_seed:num_seed + num_unlabeled]
test_files = image_files[num_seed + num_unlabeled:]

# ---------- Seed 内部再划分 ----------
num_seed_train = int(len(seed_files) * seed_train_ratio)
seed_train_files = seed_files[:num_seed_train]
seed_val_files = seed_files[num_seed_train:]

def copy_with_label(files, img_dst, label_dst):
    for img_path in files:
        label_path = labels_src / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARNING] 标签不存在: {label_path.name}")
            continue
        shutil.copy(img_path, img_dst / img_path.name)
        shutil.copy(label_path, label_dst / label_path.name)

# ---------- 拷贝 Seed ----------
copy_with_label(
    seed_train_files,
    dataset_root / "seed/images/train",
    dataset_root / "seed/labels/train",
)

copy_with_label(
    seed_val_files,
    dataset_root / "seed/images/val",
    dataset_root / "seed/labels/val",
)

# ---------- 拷贝 Unlabeled（去标注） ----------
for img_path in unlabeled_files:
    label_path = labels_src / f"{img_path.stem}.txt"
    if not label_path.exists():
        print(f"[WARNING] 标签不存在: {label_path.name}")
        continue

    shutil.copy(
        img_path,
        dataset_root / "unlabeled/images/train" / img_path.name
    )
    # 标签“隐藏”，不参与训练
    shutil.copy(
        label_path,
        dataset_root / "unlabeled/labels_hidden/train" / label_path.name
    )

# ---------- 拷贝 Test ----------
copy_with_label(
    test_files,
    dataset_root / "test/images",
    dataset_root / "test/labels",
)

# ---------- 记录划分信息 ----------
with open(dataset_root / "split_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Total images: {num_total}\n")
    f.write(f"Seed (labeled): {len(seed_files)}\n")
    f.write(f"  - Train: {len(seed_train_files)}\n")
    f.write(f"  - Val  : {len(seed_val_files)}\n")
    f.write(f"Unlabeled (simulated): {len(unlabeled_files)}\n")
    f.write(f"Test: {len(test_files)}\n")

print("===== 半自动标注数据集构建完成 =====")
print(f"Total      : {num_total}")
print(f"Seed       : {len(seed_files)}")
print(f"Unlabeled  : {len(unlabeled_files)}")
print(f"Test       : {len(test_files)}")
