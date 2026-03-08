import random
import shutil
try:
    from utils.common import PROJECT_ROOT
except ModuleNotFoundError:
    from common import PROJECT_ROOT

DATASET_ROOT = PROJECT_ROOT / "data/NEU-DET"
IMAGES_SRC = DATASET_ROOT / "images_all"
LABELS_SRC = DATASET_ROOT / "labels_all"
IMAGES_DST = DATASET_ROOT / "images"
LABELS_DST = DATASET_ROOT / "labels"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42


def copy_split(files, split):
    for img_path in files:
        label_path = LABELS_SRC / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARNING] 标签不存在: {label_path.name}")
            continue
        shutil.copy(img_path, IMAGES_DST / split / img_path.name)
        shutil.copy(label_path, LABELS_DST / split / label_path.name)


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6
    random.seed(RANDOM_SEED)

    for split in ["train", "val", "test"]:
        (IMAGES_DST / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DST / split).mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGES_SRC.glob("*.jpg")) + list(IMAGES_SRC.glob("*.png"))
    image_files.sort()
    random.shuffle(image_files)

    num_total = len(image_files)
    num_train = int(num_total * TRAIN_RATIO)
    num_val = int(num_total * VAL_RATIO)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]

    copy_split(train_files, "train")
    copy_split(val_files, "val")
    copy_split(test_files, "test")

    print("===== 数据集划分完成 =====")
    print(f"Total : {num_total}")
    print(f"Train : {len(train_files)}")
    print(f"Val   : {len(val_files)}")
    print(f"Test  : {len(test_files)}")


if __name__ == "__main__":
    main()
