import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED_ROOT = ROOT / "data/NEU-DET/seed-conservative"
PSEUDO_ROOT = ROOT / "data/NEU-DET/unlabeled-conservative"
MERGE_ROOT = ROOT / "data/NEU-DET/merge-conservative"

def collect_and_copy(images_root, labels_root):
    for img in images_root.rglob("*"):
        if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        rel_path = img.relative_to(images_root)
        lbl = labels_root / rel_path.parent / f"{img.stem}.txt"

        if lbl.exists():
            shutil.copy(img, MERGE_ROOT / "images/train" / img.name)
            shutil.copy(lbl, MERGE_ROOT / "labels/train" / lbl.name)


def main():
    if MERGE_ROOT.exists():
        shutil.rmtree(MERGE_ROOT)
    (MERGE_ROOT / "images/train").mkdir(parents=True, exist_ok=True)
    (MERGE_ROOT / "labels/train").mkdir(parents=True, exist_ok=True)

    collect_and_copy(SEED_ROOT / "images", SEED_ROOT / "labels")
    collect_and_copy(PSEUDO_ROOT / "images", PSEUDO_ROOT / "pseudo_labels")
    print("===== 合并完成 =====")


if __name__ == "__main__":
    main()
