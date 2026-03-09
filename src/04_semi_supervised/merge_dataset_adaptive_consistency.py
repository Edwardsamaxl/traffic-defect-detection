import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

seed_root = ROOT / "data/NEU-DET/seed-conservative"
pseudo_root = ROOT / "data/NEU-DET/unlabeled-conservative"
merge_root = ROOT / "data/NEU-DET/merge-adaptive-consistency"

seed = 42
random.seed(seed)

(merge_root / "images/train").mkdir(parents=True, exist_ok=True)
(merge_root / "labels/train").mkdir(parents=True, exist_ok=True)

pairs = []


def collect(img_dir, lbl_dir, repeat=1):
    for img in img_dir.glob("*"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for _ in range(repeat):
            pairs.append((img, lbl))


collect(seed_root / "images/train", seed_root / "labels/train", repeat=1)
collect(
    pseudo_root / "images/train",
    pseudo_root / "pseudo_labels_adaptive_consistency/train",
    repeat=1,
)

print(f"Total merged training samples: {len(pairs)}")
random.shuffle(pairs)

for idx, (img, lbl) in enumerate(pairs):
    new_name = f"{img.stem}_{idx}{img.suffix}"
    shutil.copy(img, merge_root / "images/train" / new_name)
    shutil.copy(lbl, merge_root / "labels/train" / f"{Path(new_name).stem}.txt")

print("===== Adaptive consistency merge 完成 =====")
print(f"Train: {len(pairs)}")
