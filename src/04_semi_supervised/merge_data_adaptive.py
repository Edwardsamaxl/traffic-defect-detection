import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

seed_root = ROOT / "data/NEU-DET/seed-conservative"
pseudo_root = ROOT / "data/NEU-DET/unlabeled-conservative"
merge_root = ROOT / "data/NEU-DET/merge-adaptive"

seed = 42
random.seed(seed)

# ---------- 创建目录 ----------
(merge_root / "images/train").mkdir(parents=True, exist_ok=True)
(merge_root / "labels/train").mkdir(parents=True, exist_ok=True)

# ---------- 收集样本 ----------
pairs = []

def collect(img_dir, lbl_dir, repeat=1):
    for img in img_dir.glob("*"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for _ in range(repeat):
            pairs.append((img, lbl))

# seed：复制 2 次（强化真实数据） / 1 次
collect(seed_root / "images/train", seed_root / "labels/train", repeat=1)

# 注意：不再使用 seed/val 参与训练

# pseudo
collect(
    pseudo_root / "images/train",
    pseudo_root / "pseudo_labels_adaptive/train",
    repeat=1
)

print(f"Total merged training samples: {len(pairs)}")

# ---------- 打乱 ----------
random.shuffle(pairs)

# ---------- 拷贝 ----------
for idx, (img, lbl) in enumerate(pairs):
    new_name = f"{img.stem}_{idx}{img.suffix}"

    shutil.copy(img, merge_root / "images/train" / new_name)
    shutil.copy(lbl, merge_root / "labels/train" / f"{Path(new_name).stem}.txt")

print("===== Adaptive merge 完成 =====")
print(f"Train: {len(pairs)}")