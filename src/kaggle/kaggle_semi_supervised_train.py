"""
Kaggle半监督训练完整脚本
=====================

使用方法:
1. 在Kaggle Notebooks中创建新Notebook
2. 将本脚本内容复制到Notebook中运行
3. 或者使用Kaggle API上传后运行

数据集: NEU-DET (需要先在Kaggle上添加数据集)
"""

# ============================================================
# 第一部分: 环境配置
# ============================================================

# 安装依赖 - editable mode
import subprocess
import sys

print("安装 ultralytics (editable mode)...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "ultralytics-main", "-q"])

import os
import yaml
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# 第二部分: 数据准备
# ============================================================

# 数据集根目录
# 优先检查 Kaggle dataset 模式，其次检查仓库本地 data 目录
# Kaggle dataset 模式: /kaggle/input/neu-det/
# 仓库clone模式: {REPO_ROOT}/data/NEU-DET/

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
    """
    划分Seed(标注数据)和Unlabeled(无标注数据)

    默认: 30%作为Seed, 70%作为Unlabeled
    """
    # 获取所有图像
    all_images = sorted([
        p for p in all_images_dir.glob("*.jpg")
    ])

    # 获取所有标签
    all_labels = {p.stem: p for p in all_labels_dir.glob("*.txt")}

    # 过滤出有标签的图像(完整数据集)
    labeled_images = [img for img in all_images if img.stem in all_labels]

    # 打乱顺序
    np.random.seed(42)
    indices = np.random.permutation(len(labeled_images))

    # 划分
    n_seed = int(len(labeled_images) * seed_ratio)
    seed_images = [labeled_images[i] for i in indices[:n_seed]]
    unlabeled_images = [labeled_images[i] for i in indices[n_seed:]]

    print(f"总图像数: {len(labeled_images)}")
    print(f"Seed (标注数据, {seed_ratio*100:.0f}%): {len(seed_images)}")
    print(f"Unlabeled (无标注数据, {(1-seed_ratio)*100:.0f}%): {len(unlabeled_images)}")

    # 创建软链接或复制
    seed_dir = OUTPUT_ROOT / "seed"
    unlabeled_dir = OUTPUT_ROOT / "unlabeled"

    # 复制Seed数据(带标签)
    for img_path in seed_images:
        # 复制图像
        dst_img = seed_dir / "images" / "train" / img_path.name
        if not dst_img.exists():
            shutil.copy(img_path, dst_img)

        # 复制标签
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

def generate_adaptive_pseudo_labels(
    model,
    unlabeled_dir: Path,
    output_dir: Path,
    baseline_ap: list,
    base_conf: float = 0.65,
    lambda_val: float = 0.25,
    iou_threshold: float = 0.6,
):
    """
    生成自适应阈值伪标签

    Args:
        model: YOLO模型
        unlabeled_dir: 无标注图像目录
        output_dir: 伪标签输出目录
        baseline_ap: 各类的AP值用于计算自适应阈值
        base_conf: 基础置信度
        lambda_val: 自适应系数
        iou_threshold: 翻转一致性IoU阈值
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算每类自适应阈值
    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)
    class_thresholds = {}

    for i, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)
        threshold = base_conf + lambda_val * (1 - norm)
        threshold = max(0.3, min(0.95, threshold))
        class_thresholds[i] = threshold

    print("各类别自适应阈值:")
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_thresholds[i]:.3f}")

    # 获取图像列表
    image_paths = sorted([
        p for p in unlabeled_dir.glob("*.jpg")
    ])

    stats = defaultdict(int)

    for img_path in image_paths:
        # 读取图像
        img = model.predict(source=str(img_path), verbose=False)[0]

        if img.boxes is None or len(img.boxes) == 0:
            continue

        valid_boxes = []
        for box in img.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            threshold = class_thresholds.get(cls_id, base_conf)

            if conf >= threshold:
                x, y, w, h = box.xywhn[0].tolist()
                valid_boxes.append((cls_id, x, y, w, h))
                stats[cls_id] += 1

        # 保存伪标签
        if valid_boxes:
            label_path = output_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("\n伪标签统计:")
    for cls_id, count in sorted(stats.items()):
        print(f"  {class_names[cls_id]}: {count}")
    print(f"总计: {sum(stats.values())} 个伪标签")

    return class_thresholds

def generate_flip_consistency_pseudo_labels(
    model,
    unlabeled_dir: Path,
    output_dir: Path,
    baseline_ap: list,
    base_conf: float = 0.65,
    lambda_val: float = 0.25,
    iou_threshold: float = 0.6,
):
    """
    生成翻转一致性伪标签(更高质量)

    同时使用自适应阈值和翻转一致性筛选
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算自适应阈值
    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)
    class_thresholds = {}

    for i, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)
        threshold = base_conf + lambda_val * (1 - norm)
        threshold = max(0.3, min(0.95, threshold))
        class_thresholds[i] = threshold

    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    print("使用翻转一致性筛选生成伪标签...")

    # 获取图像列表
    image_paths = sorted([
        p for p in unlabeled_dir.glob("*.jpg")
    ])

    stats = defaultdict(int)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 原图预测
        res_orig = model.predict(source=img, augment=False, verbose=False)[0]

        # 翻转预测
        res_flip = model.predict(source=cv2.flip(img, 1), augment=False, verbose=False)[0]

        if res_orig.boxes is None or len(res_orig.boxes) == 0:
            continue

        # 收集原图boxes
        orig_boxes = []
        for b in res_orig.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            th = class_thresholds.get(cid, base_conf)
            if conf >= th:
                orig_boxes.append((cid, conf, b.xywhn[0].tolist()))

        # 收集翻转boxes
        flip_boxes = []
        if res_flip.boxes is not None and len(res_flip.boxes) > 0:
            for b in res_flip.boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                th = class_thresholds.get(cid, base_conf)
                if conf >= th:
                    x, y, w, h = b.xywhn[0].tolist()
                    flip_boxes.append((cid, conf, [1.0 - x, y, w, h]))

        # IoU匹配
        valid_boxes = []
        for cid, conf, xywh in orig_boxes:
            matched = False

            for fcid, fconf, fxywh in flip_boxes:
                if fcid != cid:
                    continue
                # 计算IoU (简化版本)
                iou = compute_iou_simple(xywh, fxywh)
                if iou >= iou_threshold:
                    matched = True
                    break

            if matched:
                x, y, w, h = xywh
                valid_boxes.append((cid, x, y, w, h))
                stats[cid] += 1

        # 保存
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

def compute_iou_simple(box1, box2):
    """简化IoU计算 (xywh格式)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 转xyxy
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2

    # 相交区域
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + 1e-9

    return inter / union

# ============================================================
# 第四部分: 训练
# ============================================================

def train_baseline(
    model,
    data_yaml: Path,
    epochs: int = 200,
    imgsz: int = 640,
    name: str = "baseline",
):
    """训练监督学习基线"""
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=0,  # 使用GPU
        project=str(OUTPUT_ROOT / "experiments"),
        name=name,
        exist_ok=True,

        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,

        # 增强
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # 早停
        patience=50,
        close_mosaic=10,

        # 杂项
        verbose=True,
        amp=True,
    )
    return results

def train_semi_supervised(
    model,
    data_yaml: Path,
    epochs: int = 200,
    imgsz: int = 640,
    name: str = "semi_supervised",
):
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

        # 增强 - 半监督通常减少增强
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

    # 每类AP
    print(f"\n每类 mAP@0.5:")
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for key, value in metrics.results_dict.items():
        if "metrics/mAP50(" in key and key.endswith(")"):
            cls_name = key.split("(")[1].rstrip(")")
            print(f"  {cls_name}: {float(value):.4f}")

    return metrics

# ============================================================
# 主流程
# ============================================================

def main():
    """主训练流程"""

    print("="*60)
    print("Kaggle 半监督训练流程")
    print("="*60)

    # 1. 设置目录
    print("\n[1/6] 设置目录...")
    setup_directories()

    # 2. 检查数据
    print("\n[2/6] 检查数据集...")
    images_dir = DATA_ROOT / "images" / "train"
    labels_dir = DATA_ROOT / "labels" / "train"

    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")

    print(f"图像目录: {images_dir}")
    print(f"标签目录: {labels_dir}")

    # 3. 划分数据
    print("\n[3/6] 划分Seed/Unlabeled数据...")
    seed_dir, unlabeled_dir = prepare_seed_unlabeled_split(
        all_images_dir=images_dir,
        all_labels_dir=labels_dir,
        seed_ratio=0.3,
        seed_copy=3,
    )

    # 4. 导入模型
    print("\n[4/6] 加载模型...")
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")

    # 5. 训练基线获取AP
    print("\n[5/6] 训练监督学习基线...")
    # 先用Seed数据训练基线
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

    # 训练基线
    print("训练监督学习基线 (用于获取各类别AP)...")
    baseline_results = train_baseline(model, seed_yaml, epochs=100, name="baseline")

    # 获取基线AP
    print("\n获取各类别AP用于自适应阈值...")
    val_metrics = model.val(data=str(seed_yaml), split="train", verbose=False)
    baseline_ap = []
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for i in range(6):
        key = f"metrics/mAP50({class_names[i]})"
        if key in val_metrics.results_dict:
            baseline_ap.append(float(val_metrics.results_dict[key]))
        else:
            baseline_ap.append(0.5)  # 默认值

    print(f"基线AP: {dict(zip(class_names, baseline_ap))}")

    # 6. 生成伪标签
    print("\n[6/6] 生成伪标签...")

    # 创建merge数据集目录
    merge_dir = OUTPUT_ROOT / "merge"
    pseudo_dir = merge_dir / "labels"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    # 生成翻转一致性伪标签
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

    # 创建merge数据集yaml
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

    # 复制unlabeled图像到merge目录
    for img_path in (unlabeled_dir / "images").glob("*.jpg"):
        dst = merge_dir / "images" / "train" / img_path.name
        if not dst.exists():
            shutil.copy(img_path, dst)

    # 复制seed图像到merge目录
    for img_path in (seed_dir / "images" / "train").glob("*.jpg"):
        dst = merge_dir / "images" / "train" / img_path.name
        if not dst.exists():
            shutil.copy(img_path, dst)

    # 复制seed标签到merge目录
    for label_path in (seed_dir / "labels").glob("*.txt"):
        dst = merge_dir / "labels" / label_path.name
        if not dst.exists():
            shutil.copy(label_path, dst)

    print("\n伪标签生成完成!")

    # 7. 训练半监督模型
    print("\n[Extra] 训练半监督模型...")
    model_semi = YOLO(str(OUTPUT_ROOT / "experiments" / "baseline" / "weights" / "best.pt"))
    semi_results = train_semi_supervised(model_semi, merge_yaml, epochs=150, name="semi_supervised")

    # 8. 评估
    print("\n最终评估...")
    print("\n=== 基线模型 ===")
    evaluate_model(YOLO(str(OUTPUT_ROOT / "experiments" / "baseline" / "weights" / "best.pt")),
                   seed_yaml, "baseline")

    print("\n=== 半监督模型 ===")
    evaluate_model(YOLO(str(OUTPUT_ROOT / "experiments" / "semi_supervised" / "weights" / "best.pt")),
                   merge_yaml, "semi_supervised")

    print("\n" + "="*60)
    print("训练完成!")
    print(f"结果保存在: {OUTPUT_ROOT / 'experiments'}")
    print("="*60)


if __name__ == "__main__":
    main()
