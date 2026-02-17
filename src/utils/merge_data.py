import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

seed_root = ROOT / "data/NEU-DET-semi/seed"
pseudo_root = ROOT / "data/NEU-DET-semi/unlabeled"
merge_root = ROOT / "data/NEU-DET-semi/new-merge"

train_ratio = 0.8
val_ratio = 0.2
seed = 4

random.seed(seed)
assert abs(train_ratio + val_ratio - 1.0) < 1e-6

# ================== 收集所有样本 ==================
image_label_pairs = []

def collect_pairs(images_root, labels_root):
    """
    images_root: e.g. seed/images
    labels_root: e.g. seed/labels or unlabeled/pseudo_labels
    """
    for img in images_root.rglob("*"):
        if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        # 保持 train/val 子结构
        rel_path = img.relative_to(images_root)
        lbl = labels_root / rel_path.parent / f"{img.stem}.txt"

        if lbl.exists():
            image_label_pairs.append((img, lbl))

# seed（真标注）
collect_pairs(
    seed_root / "images",
    seed_root / "labels"
)

# pseudo（伪标注）
collect_pairs(
    pseudo_root / "images",
    pseudo_root / "new-pseudo_labels"
)

print(f"Total samples collected: {len(image_label_pairs)}")

if len(image_label_pairs) == 0:
    raise RuntimeError("❌ 没有收集到任何样本，请检查目录结构")

# ================== 打乱并划分 ==================
random.shuffle(image_label_pairs)

num_total = len(image_label_pairs)
num_train = int(num_total * train_ratio)

train_pairs = image_label_pairs[:num_train]
val_pairs = image_label_pairs[num_train:]

# ================== 创建目录 ==================
for split in ["train", "val"]:
    (merge_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (merge_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# ================== 拷贝 ==================
def copy_pairs(pairs, split):
    for img, lbl in pairs:
        shutil.copy(img, merge_root / "images" / split / img.name)
        shutil.copy(lbl, merge_root / "labels" / split / lbl.name)

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("===== 合并并重新划分完成 =====")
print(f"Train: {len(train_pairs)}")
print(f"Val  : {len(val_pairs)}")
