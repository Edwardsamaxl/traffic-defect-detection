"""
Kaggle半监督训练完整脚本
=====================

使用方法:
1. 先安装: !pip install -e ultralytics-main -q
2. 然后运行: !python src/kaggle/kaggle_semi_supervised_train.py
"""

import os
import yaml
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# 数据路径配置
# ============================================================
REPO_ROOT = Path("/kaggle/working/traffic-defect-detection")
DATA_ROOT = REPO_ROOT / "data" / "NEU-DET"
OUTPUT_ROOT = Path("/kaggle/working/outputs")

def setup_directories():
    dirs = [
        OUTPUT_ROOT / "seed" / "images" / "train",
        OUTPUT_ROOT / "seed" / "labels",
        OUTPUT_ROOT / "unlabeled" / "images",
        OUTPUT_ROOT / "merge" / "images" / "train",
        OUTPUT_ROOT / "merge" / "labels",
        OUTPUT_ROOT / "experiments" / "baseline",
        OUTPUT_ROOT / "experiments" / "semi_supervised",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def prepare_seed_unlabeled_split(all_images_dir, all_labels_dir, seed_ratio=0.3, seed_copy=3):
    all_images = sorted([p for p in all_images_dir.glob("*.jpg")])
    all_labels = {p.stem: p for p in all_labels_dir.glob("*.txt")}
    labeled_images = [img for img in all_images if img.stem in all_labels]

    np.random.seed(42)
    indices = np.random.permutation(len(labeled_images))
    n_seed = int(len(labeled_images) * seed_ratio)
    seed_images = [labeled_images[i] for i in indices[:n_seed]]
    unlabeled_images = [labeled_images[i] for i in indices[n_seed:]]

    print(f"总图像数: {len(labeled_images)}")
    print(f"Seed (30%): {len(seed_images)}, Unlabeled (70%): {len(unlabeled_images)}")

    seed_dir = OUTPUT_ROOT / "seed"
    unlabeled_dir = OUTPUT_ROOT / "unlabeled"

    # 复制Seed数据
    for img_path in seed_images:
        shutil.copy(img_path, seed_dir / "images" / "train" / img_path.name)
        label = all_labels.get(img_path.stem)
        if label:
            shutil.copy(label, seed_dir / "labels" / label.name)

    # 复制Unlabeled数据
    for img_path in unlabeled_images:
        shutil.copy(img_path, unlabeled_dir / "images" / img_path.name)

    # Seed复制3次
    for _ in range(seed_copy - 1):
        for img_path in seed_images:
            stem = img_path.stem
            shutil.copy(img_path, seed_dir / "images" / "train" / f"{stem}_c{_}.jpg")
            label = all_labels.get(stem)
            if label:
                shutil.copy(label, seed_dir / "labels" / f"{stem}_c{_}.txt")

    print(f"Seed数据已复制{seed_copy}次")
    return seed_dir, unlabeled_dir

def compute_iou_simple(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - inter + 1e-9
    return inter / union

def generate_pseudo_labels(model, unlabeled_dir, output_dir, baseline_ap,
                          base_conf=0.65, lambda_val=0.25, iou_threshold=0.6):
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)

    ap_min, ap_max = min(baseline_ap), max(baseline_ap)
    class_thresholds = {}
    for i, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)
        class_thresholds[i] = max(0.3, min(0.95, base_conf + lambda_val * (1 - norm)))

    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    print("自适应阈值:", {class_names[i]: f"{class_thresholds[i]:.3f}" for i in range(6)})

    image_paths = sorted([p for p in unlabeled_dir.glob("*.jpg")])
    stats = defaultdict(int)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res_orig = model.predict(source=img, augment=False, verbose=False)[0]
        res_flip = model.predict(source=cv2.flip(img, 1), augment=False, verbose=False)[0]

        if res_orig.boxes is None or len(res_orig.boxes) == 0:
            continue

        orig_boxes = []
        for b in res_orig.boxes:
            cid, conf = int(b.cls.item()), float(b.conf.item())
            if conf >= class_thresholds.get(cid, base_conf):
                orig_boxes.append((cid, conf, b.xywhn[0].tolist()))

        flip_boxes = []
        if res_flip.boxes is not None and len(res_flip.boxes) > 0:
            for b in res_flip.boxes:
                cid, conf = int(b.cls.item()), float(b.conf.item())
                if conf >= class_thresholds.get(cid, base_conf):
                    x, y, w, h = b.xywhn[0].tolist()
                    flip_boxes.append((cid, conf, [1.0 - x, y, w, h]))

        valid_boxes = []
        for cid, conf, xywh in orig_boxes:
            matched = any(
                fcid == cid and compute_iou_simple(xywh, fxywh) >= iou_threshold
                for fcid, _, fxywh in flip_boxes
            )
            if matched:
                valid_boxes.append((cid, *xywh))
                stats[cid] += 1

        if valid_boxes:
            with open(output_dir / f"{img_path.stem}.txt", "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("伪标签统计:", dict(stats))
    return stats

def train_model(model, data_yaml, epochs, name):
    return model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        device=0,
        project=str(OUTPUT_ROOT / "experiments"),
        name=name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        patience=50,
        close_mosaic=10,
        verbose=True,
        amp=True,
    )

def evaluate_model(model, data_yaml, name):
    metrics = model.val(data=str(data_yaml), split="test", imgsz=640, conf=0.001, iou=0.6, augment=True)
    print(f"\n{'='*50}")
    print(f"评估: {name}")
    print(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP@0.5: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@0.5:0.95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    return metrics

def main():
    print("="*60)
    print("Kaggle 半监督训练流程")
    print("="*60)

    print("\n[1/6] 设置目录...")
    setup_directories()

    print("\n[2/6] 检查数据集...")
    images_dir = DATA_ROOT / "images" / "train"
    labels_dir = DATA_ROOT / "labels" / "train"
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    print(f"图像目录: {images_dir}")

    print("\n[3/6] 划分数据...")
    seed_dir, unlabeled_dir = prepare_seed_unlabeled_split(images_dir, labels_dir)

    print("\n[4/6] 下载预训练权重...")
    weights_path = REPO_ROOT / "yolov8s.pt"
    if not weights_path.exists():
        print("下载 yolov8s.pt...")
        os.system(f"wget -q -O {weights_path} https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt")
    print(f"权重: {weights_path}")

    print("\n[5/6] 加载模型...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # 创建yaml
    seed_yaml = OUTPUT_ROOT / "seed.yaml"
    with open(seed_yaml, "w") as f:
        f.write(f"""path: {OUTPUT_ROOT}
train: seed/images/train
val: seed/images/train
test: {DATA_ROOT}/images/test
names:
 0: crazing
 1: inclusion
 2: patches
 3: pitted_surface
 4: rolled-in_scale
 5: scratches
""")

    print("\n[6/6] 训练监督学习基线...")
    train_model(model, seed_yaml, epochs=100, name="baseline")

    print("\n获取各类别AP...")
    val_metrics = model.val(data=str(seed_yaml), split="train", verbose=False)
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    baseline_ap = [float(val_metrics.results_dict.get(f"metrics/mAP50({cn})", 0.5)) for cn in class_names]
    print("基线AP:", dict(zip(class_names, [f"{x:.3f}" for x in baseline_ap])))

    # 生成伪标签
    print("\n生成伪标签...")
    pseudo_dir = OUTPUT_ROOT / "merge" / "labels"
    generate_pseudo_labels(model, unlabeled_dir / "images", pseudo_dir, baseline_ap)

    # 创建merge yaml
    merge_yaml = OUTPUT_ROOT / "merge.yaml"
    with open(merge_yaml, "w") as f:
        f.write(f"""path: {OUTPUT_ROOT}
train: merge/images/train
val: seed/images/train
test: {DATA_ROOT}/images/test
names:
 0: crazing
 1: inclusion
 2: patches
 3: pitted_surface
 4: rolled-in_scale
 5: scratches
""")

    # 复制数据到merge
    merge_dir = OUTPUT_ROOT / "merge"
    for img in (unlabeled_dir / "images").glob("*.jpg"):
        shutil.copy(img, merge_dir / "images" / "train" / img.name)
    for img in (seed_dir / "images" / "train").glob("*.jpg"):
        shutil.copy(img, merge_dir / "images" / "train" / img.name)
    for lbl in (seed_dir / "labels").glob("*.txt"):
        shutil.copy(lbl, merge_dir / "labels" / lbl.name)

    print("\n训练半监督模型...")
    model_semi = YOLO(str(OUTPUT_ROOT / "experiments" / "baseline" / "weights" / "best.pt"))
    train_model(model_semi, merge_yaml, epochs=150, name="semi_supervised")

    print("\n最终评估...")
    evaluate_model(YOLO(str(OUTPUT_ROOT / "experiments" / "baseline" / "weights" / "best.pt")), seed_yaml, "基线")
    evaluate_model(YOLO(str(OUTPUT_ROOT / "experiments" / "semi_supervised" / "weights" / "best.pt")), merge_yaml, "半监督")

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == "__main__":
    main()
