import cv2
import numpy as np
import random
from pathlib import Path
import shutil

# -------------------------------
# 配置路径
# -------------------------------
ROOT = Path(__file__).resolve().parents[2]

# 原始训练数据目录
TRAIN_DIR = ROOT / "data/NEU-DET"  # 原始图片和标签都在这里
IMG_DIR = TRAIN_DIR / "images/train"
LABEL_DIR = TRAIN_DIR / "labels/train"

# 增强数据输出目录（YOLOv8 期望的格式）
AUG_DIR = ROOT / "data/NEU-DET/train_copy_paste"
AUG_IMG_DIR = AUG_DIR / "images/train"   # 注意这里加了 train
AUG_LABEL_DIR = AUG_DIR / "labels/train" # 注意这里加了 train

NUM_AUG_PER_IMAGE = 2       # 每张原图生成几张增强图
MAX_PATCH_PER_IMAGE = 1     # 每张增强图最多粘贴几个 patch
SEED = 42
random.seed(SEED)

# -------------------------------
# 函数：读取 YOLO 标签
# -------------------------------
def load_labels(txt_path):
    labels = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels

# -------------------------------
# 函数：裁剪实例 patch
# -------------------------------
def crop_instance(image, bbox):
    h, w = image.shape[:2]
    cls, xc, yc, bw, bh = bbox
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    patch = image[y1:y2, x1:x2].copy()
    return patch, cls, (x2-x1, y2-y1)

# -------------------------------
# 函数：粘贴 patch 到目标图
# -------------------------------
def paste_instance(target_img, patch):
    th, tw = target_img.shape[:2]
    ph, pw = patch.shape[:2]

    if tw - pw <= 0 or th - ph <= 0:
        return target_img, None

    x = random.randint(0, tw - pw)
    y = random.randint(0, th - ph)

    target_img[y:y+ph, x:x+pw] = patch

    xc = (x + pw/2) / tw
    yc = (y + ph/2) / th
    bw = pw / tw
    bh = ph / th

    return target_img, [xc, yc, bw, bh]

# -------------------------------
# 生成增强数据集
# -------------------------------
def generate_aug_dataset(img_dir, label_dir, aug_img_dir, aug_label_dir):
    # 删除旧目录
    if aug_img_dir.exists():
        shutil.rmtree(aug_img_dir)
    if aug_label_dir.exists():
        shutil.rmtree(aug_label_dir)
    # 创建目录
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_label_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(img_dir.glob("*.jpg"))

    for img_file in img_files:
        txt_file = label_dir / f"{img_file.stem}.txt"
        if not txt_file.exists():
            continue
        labels = load_labels(txt_file)
        image = cv2.imread(str(img_file))

        # 保存原图
        shutil.copy(img_file, aug_img_dir / img_file.name)
        shutil.copy(txt_file, aug_label_dir / f"{img_file.stem}.txt")

        for n in range(NUM_AUG_PER_IMAGE):
            aug_image = image.copy()
            aug_labels = labels.copy()

            for _ in range(MAX_PATCH_PER_IMAGE):
                src_file = random.choice(img_files)
                src_txt = label_dir / f"{src_file.stem}.txt"
                src_labels = load_labels(src_txt)
                if not src_labels:
                    continue
                src_bbox = random.choice(src_labels)
                src_image = cv2.imread(str(src_file))
                patch, cls, _ = crop_instance(src_image, src_bbox)

                aug_image, new_bbox = paste_instance(aug_image, patch)
                if new_bbox is not None:
                    aug_labels.append([cls] + new_bbox)

            # 保存增强图片
            aug_name = img_file.stem + f"_aug{n}.jpg"
            cv2.imwrite(str(aug_img_dir / aug_name), aug_image)

            # 保存增强标签
            aug_txt_name = img_file.stem + f"_aug{n}.txt"
            with open(aug_label_dir / aug_txt_name, "w") as f:
                for bbox in aug_labels:
                    line = f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
                    f.write(line)

    print(f"===== 离线 Copy-Paste 增强完成 =====")
    print(f"图片目录: {aug_img_dir}")
    print(f"标签目录: {aug_label_dir}")

# -------------------------------
# 执行
# -------------------------------
if __name__ == "__main__":
    generate_aug_dataset(IMG_DIR, LABEL_DIR, AUG_IMG_DIR, AUG_LABEL_DIR)