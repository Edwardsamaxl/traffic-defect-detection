import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

seed_root = ROOT / "data/NEU-DET/seed"
pseudo_root = ROOT / "data/NEU-DET/unlabeled"
merge_root = ROOT / "data/NEU-DET/merge"

# 删除旧 merge（防止污染）
if merge_root.exists():
    shutil.rmtree(merge_root)

(merge_root / "images/train").mkdir(parents=True, exist_ok=True)
(merge_root / "labels/train").mkdir(parents=True, exist_ok=True)

def collect_and_copy(images_root, labels_root):
    for img in images_root.rglob("*"):
        if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        rel_path = img.relative_to(images_root)
        lbl = labels_root / rel_path.parent / f"{img.stem}.txt"

        if lbl.exists():
            shutil.copy(img, merge_root / "images/train" / img.name)
            shutil.copy(lbl, merge_root / "labels/train" / lbl.name)

# 复制 seed
collect_and_copy(
    seed_root / "images",
    seed_root / "labels"
)

# 复制 pseudo
collect_and_copy(
    pseudo_root / "images",
    pseudo_root / "pseudo_labels"
)

print("===== 合并完成 =====")
