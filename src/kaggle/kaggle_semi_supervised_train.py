"""
Kaggle半监督训练完整脚本
=====================

使用方法:
1. 先运行安装命令: !pip install -e ultralytics-main -q
2. 然后运行本脚本: %run src/kaggle/kaggle_semi_supervised_train.py

数据集: NEU-DET (需要先在Kaggle上添加数据集)
"""

# ============================================================
# 第一部分: 环境配置
# ============================================================
# 注意: 先运行以下命令安装ultralytics:
# !pip install -e ultralytics-main -q

import yaml
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# 第二部分: 数据准备
# ============================================================

# 数据集根目录
REPO_ROOT = Path("/kaggle/working/traffic-defect-detection")
KAGGLE_INPUT = Path("/kaggle/input/neu-det")
LOCAL_DATA = REPO_ROOT / "data" / "NEU-DET"

# 自动选择数据路径
if KAGGLE_INPUT.exists():
    DATA_ROOT = KAGGLE_INPUT
elif LOCAL_DATA.exists():
    DATA_ROOT = LOCAL_DATA
else:
    raise FileNotFoundError(f"数据集未找到: {KAGGLE_INPUT} 或 {LOCAL_DATA}")

OUTPUT_ROOT = Path("/kaggle/working/outputs")

def setup_directories():
    """创建必要的目录结构"""
    dirs = [
        OUTPUT_ROOT / "seed" / "images" / "train",
        OUTPUT_ROOT / "seed" / "labels",
        OUTPUT_ROOT / "unlabeled" / "images",
        OUTPUT_ROOT / "merge" / "images" / "train",
        OUTPUT_ROOT / "merge" / "labels",
        OUTPUT_ROOT / "experiments" / "baseline",
        OUTPUT_ROOT / "experiments" / "semi_supervised",
        OUTPUT_ROOT / "experiments" / "semi_full",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def prepare_seed_unlabeled_split(
    all_images_dir: Path,
    all_labels_dir: Path,
    seed_ratio: float = 0.3,
    seed_copy: int = 3,
):
    """划分Seed(标注数据)和Unlabeled(无标注数据)"""
    all_images = sorted([p for p in all_images_dir.glob("*.jpg")])
    all_labels = {p.stem: p for p in all_labels_dir.glob("*.txt")}
    labeled_images = [img for img in all_images if img.stem in all_labels]

    np.random.seed(42)
    indices = np.random.permutation(len(labeled_images))
    n_seed = int(len(labeled_images) * seed_ratio)
    seed_images = [labeled_images[i] for i in indices[:n_seed]]
    unlabeled_images = [labeled_images[i] for i in indices[n_seed:]]

    print(f"总图像数: {len(labeled_images)}")
    print(f"Seed (标注数据, {seed_ratio*100:.0f}%): {len(seed_images)}")
    print(f"Unlabeled (无标注数据, {(1-seed_ratio)*100:.0f}%): {len(unlabeled_images)}")

    seed_dir = OUTPUT_ROOT / "seed"
    unlabeled_dir = OUTPUT_ROOT / "unlabeled"

    # 复制Seed数据(带标签)
    for img_path in seed_images:
        dst_img = seed_dir / "images" / "train" / img_path.name
        if not dst_img.exists():
            shutil.copy(img_path, dst_img)
        label_path = all_labels.get(img_path.stem)
        if label_path:
            dst_label = seed_dir / "labels" / label_path.name
            if not dst_label.exists():
                shutil.copy(label_path, dst_label)

    # 复制Unlabeled数据(只复制图像，不复制标签)
    for img_path in unlabeled_images:
        dst_img = unlabeled_dir / "images" / img_path.name
        if not dst_img.exists():
            shutil.copy(img_path, dst_img)

    # Seed数据复制3次(降低伪标签权重)
    for _ in range(seed_copy - 1):
        for img_path in seed_images:
            stem = img_path.stem
            dst_img = seed_dir / "images" / "train" / f"{stem}_copy{_}.jpg"
            if not dst_img.exists():
                shutil.copy(img_path, dst_img)
            label_path = all_labels.get(stem)
            if label_path:
                dst_label = seed_dir / "labels" / f"{stem}_copy{_}.txt"
                if not dst_label.exists():
                    shutil.copy(label_path, dst_label)

    print(f"Seed数据已复制{seed_copy}次用于强化")
    return seed_dir, unlabeled_dir

# ============================================================
# 第三部分: 伪标签生成
# ============================================================

def compute_iou_simple(box1, box2):
    """简化IoU计算 (xywh格式)"""
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

def generate_flip_consistency_pseudo_labels(
    model,
    unlabeled_dir: Path,
    output_dir: Path,
    baseline_ap: list,
    base_conf: float = 0.65,
    lambda_val: float = 0.25,
    iou_threshold: float = 0.6,
):
    """生成翻转一致性伪标签"""
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)

    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)
    class_thresholds = {}
    for i, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)
        threshold = base_conf + lambda_val * (1 - norm)
        class_thresholds[i] = max(0.3, min(0.95, threshold))

    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    print("各类别自适应阈值:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_thresholds[i]:.3f}")

    print("使用翻转一致性筛选生成伪标签...")
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
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            th = class_thresholds.get(cid, base_conf)
            if conf >= th:
                orig_boxes.append((cid, conf, b.xywhn[0].tolist()))

        flip_boxes = []
        if res_flip.boxes is not None and len(res_flip.boxes) > 0:
            for b in res_flip.boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                th = class_thresholds.get(cid, base_conf)
                if conf >= th:
                    x, y, w, h = b.xywhn[0].tolist()
                    flip_boxes.append((cid, conf, [1.0 - x, y, w, h]))

        valid_boxes = []
        for cid, conf, xywh in orig_boxes:
            matched = False
            for fcid, fconf, fxywh in flip_boxes:
                if fcid != cid:
                    continue
                if compute_iou_simple(xywh, fxywh) >= iou_threshold:
                    matched = True
                    break
            if matched:
                x, y, w, h = xywh
                valid_boxes.append((cid, x, y, w, h))
                stats[cid] += 1

        if valid_boxes:
            label_path = output_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("\n翻转一致性伪标签统计:")
    for cls_id, count in sorted(stats.items()):
        print(f"  {class_names[cls_id]}: {count}")
    print(f"总计: {sum(stats.values())} 个伪标签")
    return stats

# ============================================================
# 第四部分: 训练
# ============================================================

def train_baseline(model, data_yaml: Path, epochs: int = 200, imgsz: int = 640, name: str = "baseline"):
    """训练监督学习基线"""
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
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
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=50,
        close_mosaic=10,
        verbose=True,
        amp=True,
    )
    return results

def train_semi_supervised(model, data_yaml: Path, epochs: int = 200, imgsz: int = 640, name: str = "semi_supervised"):
    """训练半监督模型"""
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
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
        flipud=0.3,
        fliplr=0.5,
        patience=50,
        close_mosaic=10,
        verbose=True,
        amp=True,
    )
    return results

# ============================================================
# 第五部分: 评估
# ============================================================

def evaluate_model(model, data_yaml: Path, name: str = "eval"):
    """评估模型"""
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.6,
        augment=True,
    )
    print(f"\n{'='*50}")
    print(f"评估结果: {name}")
    print(f"{'='*50}")
    print(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall:    {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP@0.5:   {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@0.5:0.95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    return metrics

# ============================================================
# 主流程
# ============================================================

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
    print(f"标签目录: {labels_dir}")

    print("\n[3/6] 划分Seed/Unlabeled数据...")
    seed_dir, unlabeled_dir = prepare_seed_unlabeled_split(
        all_images_dir=images_dir,
        all_labels_dir=labels_dir,
        seed_ratio=0.3,
        seed_copy=3,
    )

    print("\n[4/6] 下载预训练权重...")
    import urllib.request
    weights_path = REPO_ROOT / "yolov8s.pt"
    if not weights_path.exists():
        print("下载 yolov8s.pt...")
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            str(weights_path)
        )
        print(f"已下载到: {weights_path}")

    print("\n[5/6] 加载模型...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    print("\n[5/6] 训练监督学习基线...")
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

    print("训练监督学习基线 (用于获取各类别AP)...")
    train_baseline(model, seed_yaml, epochs=100, name="baseline")

    print("\n获取各类别AP用于自适应阈值...")
    val_metrics = model.val(data=str(seed_yaml), split="train", verbose=False)
    baseline_ap = []
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for i in range(6):
        key = f"metrics/mAP50({class_names[i]})"
        if key in val_metrics.results_dict:
            baseline_ap.append(float(val_metrics.results_dict[key]))
        else:
            baseline_ap.append(0.5)
    print(f"基线AP: {dict(zip(class_names, baseline_ap))}")

    print("\n[6/6] 生成伪标签...")
    merge_dir = OUTPUT_ROOT / "merge"
    pseudo_dir = merge_dir / "labels"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    print("生成翻转一致性伪标签...")
    generate_flip_consistency_pseudo_labels(
        model=model,
        unlabeled_dir=unlabeled_dir / "images",
        output_dir=pseudo_dir,
        baseline_ap=baseline_ap,
        base_conf=0.65,
        lambda_val=0.25,
        iou_threshold=0.6,
    )

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

    # 复制数据到merge目录
    for img_path in (unlabeled_dir / "images").glob("*.jpg"):
        dst = merge_dir / "images" / "train" / img_path.name
        if not dst.exists():
            shutil.copy(img_path, dst)
    for img_path in (seed_dir / "images" / "train").glob("*.jpg"):
        dst = merge_dir / "images" / "train" / img_path.name
        if not dst.exists():
            shutil.copy(img_path, dst)
    for label_path in (seed_dir / "labels").glob("*.txt"):
        dst = merge_dir / "labels" / label_path.name
        if not dst.exists():
            shutil.copy(label_path, dst)

    print("\n伪标签生成完成!")

    print("\n[Extra] 训练半监督模型...")
    model_semi = YOLO(str(OUTPUT_ROOT / "experiments" / "baseline" / "weights" / "best.pt"))
    train_semi_supervised(model_semi, merge_yaml, epochs=150, name="semi_supervised")

    print("\n最终评估...")
    print("\n=== 基线模型 ===")
    evaluate_model(YOLO(str(OUTPUT_ROOT / "experiments" / "baseline" / "weights" / "best.pt")), seed_yaml, "baseline")
    print("\n=== 半监督模型 ===")
    evaluate_model(YOLO(str(OUTPUT_ROOT / "experiments" / "semi_supervised" / "weights" / "best.pt")), merge_yaml, "semi_supervised")

    print("\n" + "="*60)
    print("训练完成!")
    print(f"结果保存在: {OUTPUT_ROOT / 'experiments'}")
    print("="*60)

if __name__ == "__main__":
    main()
