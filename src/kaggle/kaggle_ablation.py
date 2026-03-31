"""
Kaggle消融实验脚本
=================

用于验证各个组件的贡献:
1. Baseline (纯监督)
2. + Standard Pseudo-labels (固定阈值0.7)
3. + Adaptive Threshold (自适应阈值)
4. + Flip Consistency (翻转一致性)
5. Full (自适应 + 一致性)

使用方法: 复制到Kaggle Notebook运行
"""

import os
import yaml
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

DATA_ROOT = Path("/kaggle/input/neu-det")
OUTPUT_ROOT = Path("/kaggle/working/outputs")

# 缺陷类别
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)

# 实验配置
ABLATION_CONFIGS = {
    "ablation1_baseline": {
        "description": "纯监督学习基线",
        "use_pseudo_labels": False,
        "pseudo_label_type": None,
    },
    "ablation2_std_pseudo": {
        "description": "标准伪标签 (固定阈值0.7)",
        "use_pseudo_labels": True,
        "pseudo_label_type": "standard",
        "conf_threshold": 0.7,
    },
    "ablation3_adaptive": {
        "description": "自适应阈值伪标签",
        "use_pseudo_labels": True,
        "pseudo_label_type": "adaptive",
        "base_conf": 0.65,
        "lambda_val": 0.25,
    },
    "ablation4_consistency": {
        "description": "翻转一致性筛选",
        "use_pseudo_labels": True,
        "pseudo_label_type": "flip_consistency",
        "base_conf": 0.65,
        "lambda_val": 0.25,
        "iou_threshold": 0.6,
    },
}

# ============================================================
# 辅助函数
# ============================================================

def setup_directories():
    """创建目录结构"""
    dirs = [
        OUTPUT_ROOT / "seed" / "images" / "train",
        OUTPUT_ROOT / "seed" / "labels",
        OUTPUT_ROOT / "unlabeled" / "images",
        OUTPUT_ROOT / "merge" / "images" / "train",
        OUTPUT_ROOT / "merge" / "labels",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def prepare_data(seed_ratio=0.3, seed_copy=3):
    """准备数据划分"""
    images_dir = DATA_ROOT / "images" / "train"
    labels_dir = DATA_ROOT / "labels" / "train"

    # 获取所有有标签的图像
    all_images = [p for p in images_dir.glob("*.jpg")]
    all_labels = {p.stem: p for p in labels_dir.glob("*.txt")}
    labeled_images = [img for img in all_images if img.stem in all_labels]

    # 划分
    np.random.seed(42)
    indices = np.random.permutation(len(labeled_images))
    n_seed = int(len(labeled_images) * seed_ratio)

    seed_images = [labeled_images[i] for i in indices[:n_seed]]
    unlabeled_images = [labeled_images[i] for i in indices[n_seed:]]

    # 复制Seed数据
    seed_dir = OUTPUT_ROOT / "seed"
    for img_path in seed_images:
        shutil.copy(img_path, seed_dir / "images" / "train" / img_path.name)
        label = all_labels.get(img_path.stem)
        if label:
            shutil.copy(label, seed_dir / "labels" / label.name)

    # Seed复制强化
    for _ in range(seed_copy - 1):
        for img_path in seed_images:
            stem = img_path.stem
            shutil.copy(img_path, seed_dir / "images" / "train" / f"{stem}_c{_}.jpg")
            label = all_labels.get(stem)
            if label:
                shutil.copy(label, seed_dir / "labels" / f"{stem}_c{_}.txt")

    # 复制Unlabeled数据(无标签)
    unlabeled_dir = OUTPUT_ROOT / "unlabeled"
    for img_path in unlabeled_images:
        shutil.copy(img_path, unlabeled_dir / "images" / img_path.name)

    return seed_dir, unlabeled_dir

def compute_iou_simple(box1, box2):
    """IoU计算 (xywh格式)"""
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

def generate_pseudo_labels(model, config):
    """生成伪标签"""
    import cv2

    unlabeled_dir = OUTPUT_ROOT / "unlabeled"
    pseudo_dir = OUTPUT_ROOT / "merge" / "labels"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    ptype = config["pseudo_label_type"]

    if ptype == "standard":
        # 标准伪标签 - 固定阈值
        conf_th = config.get("conf_threshold", 0.7)
        image_paths = sorted(unlabeled_dir.glob("*.jpg"))

        for img_path in image_paths:
            result = model.predict(source=str(img_path), verbose=False)[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            valid_boxes = []
            for box in result.boxes:
                if float(box.conf.item()) >= conf_th:
                    xywh = box.xywhn[0].tolist()
                    valid_boxes.append((int(box.cls.item()), *xywh))

            if valid_boxes:
                with open(pseudo_dir / f"{img_path.stem}.txt", "w") as f:
                    for cls_id, x, y, w, h in valid_boxes:
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    elif ptype == "adaptive":
        # 自适应阈值
        base_conf = config.get("base_conf", 0.65)
        lambda_val = config.get("lambda_val", 0.25)

        # 获取baseline AP
        seed_yaml = OUTPUT_ROOT / "seed.yaml"
        val_metrics = model.val(data=str(seed_yaml), split="train", verbose=False)

        baseline_ap = []
        for i in range(NUM_CLASSES):
            key = f"metrics/mAP50({CLASSES[i]})"
            baseline_ap.append(float(val_metrics.results_dict.get(key, 0.5)))

        # 计算自适应阈值
        ap_min, ap_max = min(baseline_ap), max(baseline_ap)
        class_thresholds = {}
        for i, ap in enumerate(baseline_ap):
            norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)
            threshold = base_conf + lambda_val * (1 - norm)
            class_thresholds[i] = max(0.3, min(0.95, threshold))

        # 生成伪标签
        image_paths = sorted(unlabeled_dir.glob("*.jpg"))
        for img_path in image_paths:
            result = model.predict(source=str(img_path), verbose=False)[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            valid_boxes = []
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                threshold = class_thresholds.get(cls_id, base_conf)

                if conf >= threshold:
                    xywh = box.xywhn[0].tolist()
                    valid_boxes.append((cls_id, *xywh))

            if valid_boxes:
                with open(pseudo_dir / f"{img_path.stem}.txt", "w") as f:
                    for cls_id, x, y, w, h in valid_boxes:
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    elif ptype == "flip_consistency":
        # 翻转一致性
        base_conf = config.get("base_conf", 0.65)
        lambda_val = config.get("lambda_val", 0.25)
        iou_th = config.get("iou_threshold", 0.6)

        # 获取baseline AP和阈值
        seed_yaml = OUTPUT_ROOT / "seed.yaml"
        val_metrics = model.val(data=str(seed_yaml), split="train", verbose=False)

        baseline_ap = []
        for i in range(NUM_CLASSES):
            key = f"metrics/mAP50({CLASSES[i]})"
            baseline_ap.append(float(val_metrics.results_dict.get(key, 0.5)))

        ap_min, ap_max = min(baseline_ap), max(baseline_ap)
        class_thresholds = {}
        for i, ap in enumerate(baseline_ap):
            norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)
            threshold = base_conf + lambda_val * (1 - norm)
            class_thresholds[i] = max(0.3, min(0.95, threshold))

        # 生成伪标签
        image_paths = sorted(unlabeled_dir.glob("*.jpg"))
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            res_orig = model.predict(source=img, augment=False, verbose=False)[0]
            res_flip = model.predict(source=cv2.flip(img, 1), augment=False, verbose=False)[0]

            if res_orig.boxes is None or len(res_orig.boxes) == 0:
                continue

            # 收集原图boxes
            orig_boxes = []
            for b in res_orig.boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                if conf >= class_thresholds.get(cid, base_conf):
                    orig_boxes.append((cid, conf, b.xywhn[0].tolist()))

            # 收集翻转boxes
            flip_boxes = []
            if res_flip.boxes is not None and len(res_flip.boxes) > 0:
                for b in res_flip.boxes:
                    cid = int(b.cls.item())
                    conf = float(b.conf.item())
                    if conf >= class_thresholds.get(cid, base_conf):
                        x, y, w, h = b.xywhn[0].tolist()
                        flip_boxes.append((cid, conf, [1.0 - x, y, w, h]))

            # 匹配
            valid_boxes = []
            for cid, conf, xywh in orig_boxes:
                matched = False
                for fcid, fconf, fxywh in flip_boxes:
                    if fcid != cid:
                        continue
                    if compute_iou_simple(xywh, fxywh) >= iou_th:
                        matched = True
                        break
                if matched:
                    valid_boxes.append((cid, *xywh))

            if valid_boxes:
                with open(pseudo_dir / f"{img_path.stem}.txt", "w") as f:
                    for cls_id, x, y, w, h in valid_boxes:
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def create_yaml_files():
    """创建数据集yaml文件"""
    # Seed yaml
    seed_yaml_content = f"""path: {OUTPUT_ROOT}
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
"""
    with open(OUTPUT_ROOT / "seed.yaml", "w") as f:
        f.write(seed_yaml_content)

    # Merge yaml
    merge_yaml_content = f"""path: {OUTPUT_ROOT}
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
"""
    with open(OUTPUT_ROOT / "merge.yaml", "w") as f:
        f.write(merge_yaml_content)

def prepare_merge_dataset():
    """准备merge数据集"""
    seed_dir = OUTPUT_ROOT / "seed"
    unlabeled_dir = OUTPUT_ROOT / "unlabeled"
    merge_dir = OUTPUT_ROOT / "merge"
    pseudo_dir = merge_dir / "labels"

    # 复制seed图像
    for img_path in (seed_dir / "images" / "train").glob("*.jpg"):
        dst = merge_dir / "images" / "train" / img_path.name
        if not dst.exists():
            shutil.copy(img_path, dst)

    # 复制seed标签
    for label_path in (seed_dir / "labels").glob("*.txt"):
        dst = merge_dir / "labels" / label_path.name
        if not dst.exists():
            shutil.copy(label_path, dst)

    # 复制unlabeled图像和伪标签
    for img_path in (unlabeled_dir / "images").glob("*.jpg"):
        dst = merge_dir / "images" / "train" / img_path.name
        if not dst.exists():
            shutil.copy(img_path, dst)

    # 伪标签已在 generate_pseudo_labels 中生成

def train_model(model, data_yaml, epochs, name):
    """训练模型"""
    results = model.train(
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
        flipud=0.5,
        fliplr=0.5,
        patience=50,
        close_mosaic=10,
        verbose=True,
        amp=True,
    )
    return results

def evaluate(model, data_yaml, name):
    """评估模型"""
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.6,
        augment=True,
    )

    results = {
        "name": name,
        "precision": float(metrics.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(metrics.results_dict.get("metrics/recall(B)", 0)),
        "map50": float(metrics.results_dict.get("metrics/mAP50(B)", 0)),
        "map50_95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0)),
    }

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"mAP@0.5:    {results['map50']:.4f}")
    print(f"mAP@0.5:0.95: {results['map50_95']:.4f}")

    return results

# ============================================================
# 主流程
# ============================================================

def main():
    from ultralytics import YOLO

    print("="*60)
    print("消融实验")
    print("="*60)

    # 1. 设置
    print("\n[1/5] 设置目录...")
    setup_directories()
    create_yaml_files()

    # 2. 准备数据
    print("\n[2/5] 准备数据...")
    prepare_data(seed_ratio=0.3, seed_copy=3)

    # 3. 训练基线
    print("\n[3/5] 训练监督学习基线...")
    model = YOLO("yolov8s.pt")
    train_model(model, OUTPUT_ROOT / "seed.yaml", epochs=100, name="ablation1_baseline")

    # 4. 获取基线AP
    print("\n获取基线各类别AP...")
    val_metrics = model.val(data=str(OUTPUT_ROOT / "seed.yaml"), split="train", verbose=False)
    baseline_ap = []
    for i in range(NUM_CLASSES):
        key = f"metrics/mAP50({CLASSES[i]})"
        baseline_ap.append(float(val_metrics.results_dict.get(key, 0.5)))
    print(f"Baseline AP: {dict(zip(CLASSES, baseline_ap))}")

    # 5. 逐个消融实验
    print("\n[4/5] 运行消融实验...")

    results_all = []

    # 实验1: 基线
    print("\n--- 实验1: 纯监督基线 ---")
    model_baseline = YOLO(str(OUTPUT_ROOT / "experiments" / "ablation1_baseline" / "weights" / "best.pt"))
    r = evaluate(model_baseline, OUTPUT_ROOT / "seed.yaml", "Baseline (Supervised)")
    results_all.append(r)

    # 准备merge数据集(无伪标签版本 - 只需要seed数据)
    # 已经有了seed数据

    # 实验2: 标准伪标签
    print("\n--- 实验2: 标准伪标签 ---")
    config = ABLATION_CONFIGS["ablation2_std_pseudo"]

    # 生成标准伪标签
    generate_pseudo_labels(model_baseline, config)
    prepare_merge_dataset()

    # 训练
    model_std = YOLO(str(OUTPUT_ROOT / "experiments" / "ablation1_baseline" / "weights" / "best.pt"))
    train_model(model_std, OUTPUT_ROOT / "merge.yaml", epochs=100, name="ablation2_std_pseudo")
    r = evaluate(model_std, OUTPUT_ROOT / "merge.yaml", "Standard Pseudo-labels")
    results_all.append(r)

    # 实验3: 自适应阈值
    print("\n--- 实验3: 自适应阈值 ---")
    config = ABLATION_CONFIGS["ablation3_adaptive"]

    # 清空旧伪标签
    pseudo_dir = OUTPUT_ROOT / "merge" / "labels"
    for f in pseudo_dir.glob("*.txt"):
        if "crazing" not in f.name:  # 保留seed标签
            f.unlink()

    generate_pseudo_labels(model_baseline, config)
    prepare_merge_dataset()

    model_adaptive = YOLO(str(OUTPUT_ROOT / "experiments" / "ablation1_baseline" / "weights" / "best.pt"))
    train_model(model_adaptive, OUTPUT_ROOT / "merge.yaml", epochs=100, name="ablation3_adaptive")
    r = evaluate(model_adaptive, OUTPUT_ROOT / "merge.yaml", "Adaptive Threshold")
    results_all.append(r)

    # 实验4: 翻转一致性
    print("\n--- 实验4: 翻转一致性 ---")
    config = ABLATION_CONFIGS["ablation4_consistency"]

    # 清空旧伪标签
    for f in pseudo_dir.glob("*.txt"):
        if "crazing" not in f.name:
            f.unlink()

    generate_pseudo_labels(model_baseline, config)
    prepare_merge_dataset()

    model_consistency = YOLO(str(OUTPUT_ROOT / "experiments" / "ablation1_baseline" / "weights" / "best.pt"))
    train_model(model_consistency, OUTPUT_ROOT / "merge.yaml", epochs=100, name="ablation4_consistency")
    r = evaluate(model_consistency, OUTPUT_ROOT / "merge.yaml", "Flip Consistency")
    results_all.append(r)

    # 6. 保存结果
    print("\n[5/5] 保存结果...")

    print("\n" + "="*60)
    print("消融实验汇总")
    print("="*60)
    print(f"{'Configuration':<30} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
    print("-" * 80)
    for r in results_all:
        print(f"{r['name']:<30} {r['precision']:<12.4f} {r['recall']:<12.4f} "
              f"{r['map50']:<12.4f} {r['map50_95']:<12.4f}")

    # 保存为CSV
    import csv
    csv_path = OUTPUT_ROOT / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "precision", "recall", "map50", "map50_95"])
        writer.writeheader()
        writer.writerows(results_all)

    print(f"\n结果已保存: {csv_path}")

if __name__ == "__main__":
    main()
