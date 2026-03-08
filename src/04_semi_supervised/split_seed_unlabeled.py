import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/NEU-DET"
IMAGES_SRC = DATASET_ROOT / "images/train"
LABELS_SRC = DATASET_ROOT / "labels/train"
SEED_RATIO = 0.2
UNLABELED_RATIO = 0.8
RANDOM_SEED = 42


def main():
    assert abs(SEED_RATIO + UNLABELED_RATIO - 1.0) < 1e-6
    random.seed(RANDOM_SEED)

    output_dirs = [
        DATASET_ROOT / "seed/images/train",
        DATASET_ROOT / "seed/labels/train",
        DATASET_ROOT / "unlabeled/images/train",
        DATASET_ROOT / "unlabeled/labels_hidden/train",
    ]
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGES_SRC.glob("*.jpg")) + list(IMAGES_SRC.glob("*.png"))
    image_files.sort()
    random.shuffle(image_files)

    num_total = len(image_files)
    num_seed = int(num_total * SEED_RATIO)
    seed_files = image_files[:num_seed]
    unlabeled_files = image_files[num_seed:]

    for img_path in seed_files:
        label_path = LABELS_SRC / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARNING] 标签不存在: {label_path.name}")
            continue
        shutil.copy(img_path, DATASET_ROOT / "seed/images/train" / img_path.name)
        shutil.copy(label_path, DATASET_ROOT / "seed/labels/train" / label_path.name)

    for img_path in unlabeled_files:
        label_path = LABELS_SRC / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARNING] 标签不存在: {label_path.name}")
            continue
        shutil.copy(img_path, DATASET_ROOT / "unlabeled/images/train" / img_path.name)
        shutil.copy(label_path, DATASET_ROOT / "unlabeled/labels_hidden/train" / label_path.name)

    with open(DATASET_ROOT / "split_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Total images: {num_total}\n")
        f.write(f"Seed (labeled): {len(seed_files)}\n")
        f.write(f"Unlabeled (simulated): {len(unlabeled_files)}\n")

    print("===== 半监督数据集构建完成 =====")
    print(f"Total      : {num_total}")
    print(f"Seed       : {len(seed_files)}")
    print(f"Unlabeled  : {len(unlabeled_files)}")


if __name__ == "__main__":
    main()
