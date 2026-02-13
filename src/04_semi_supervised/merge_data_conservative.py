import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

seed_root = ROOT / "data/NEU-DET-semi/seed"
pseudo_root = ROOT / "data/NEU-DET-semi/unlabeled"
merge_root = ROOT / "data/NEU-DET-semi/merge-conservative"

train_ratio = 0.8
seed = 42
random.seed(seed)

assert abs(train_ratio - 0.8) < 1e-6

# ---------- 创建目录 ----------
for split in ["train", "val"]:
    (merge_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (merge_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# ---------- 收集样本 ----------
pairs = []

def collect(img_dir, lbl_dir, repeat=1):
    for img in img_dir.glob("*"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for _ in range(repeat):
            pairs.append((img, lbl))

# seed：复制 1 份（共 2 次）
collect(seed_root / "images/train", seed_root / "labels/train", repeat=2)
collect(seed_root / "images/val",   seed_root / "labels/val",   repeat=2)

# pseudo：只用 pseudo_labels
collect(
    pseudo_root / "images/train",
    pseudo_root / "pseudo_labels-conservative/train",
    repeat=1
)

print(f"Total merged samples (with duplication): {len(pairs)}")

# ---------- 打乱并重新划分 ----------
random.shuffle(pairs)

num_total = len(pairs)
num_train = int(num_total * train_ratio)

train_pairs = pairs[:num_train]
val_pairs   = pairs[num_train:]

# ---------- 拷贝 ----------
def copy_pairs(pairs, split):
    for idx, (img, lbl) in enumerate(pairs):
        new_name = f"{img.stem}_{idx}{img.suffix}"
        shutil.copy(img, merge_root / "images" / split / new_name)
        shutil.copy(lbl, merge_root / "labels" / split / f"{Path(new_name).stem}.txt")

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("===== Conservative merge 完成 =====")
print(f"Train: {len(train_pairs)}")
print(f"Val  : {len(val_pairs)}")
