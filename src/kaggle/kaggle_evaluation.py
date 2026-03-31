"""
Kaggle模型评估脚本
=================

用于评估训练好的模型，生成详细的评估报告

使用方法:
1. 将权重文件上传到 Kaggle
2. 修改下面的 WEIGHTS_PATH
3. 运行脚本
"""

import os
import yaml
import csv
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================

# 数据集路径
DATA_ROOT = Path("/kaggle/input/neu-det")
OUTPUT_ROOT = Path("/kaggle/working/outputs")

# 评估的模型列表
# 格式: (name, weight_path)
MODELS_TO_EVALUATE = [
    ("Baseline (Supervised)", "/kaggle/working/outputs/experiments/baseline/weights/best.pt"),
    ("Semi-supervised", "/kaggle/working/outputs/experiments/semi_supervised/weights/best.pt"),
    # 添加更多模型...
]

# 测试集配置
TEST_DATA_YAML = """path: /kaggle/input/neu-det
train: images/train
val: images/val
test: images/test

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""

# 缺陷类别
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)

# ============================================================
# 评估函数
# ============================================================

def create_test_yaml():
    """创建测试集yaml"""
    yaml_path = OUTPUT_ROOT / "test.yaml"
    with open(yaml_path, "w") as f:
        f.write(TEST_DATA_YAML)
    return yaml_path

def evaluate_model(model, weight_path: Path, test_yaml: Path, name: str):
    """
    评估单个模型

    Returns:
        dict: 包含各项指标的字典
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"评估: {name}")
    print(f"权重: {weight_path}")
    print(f"{'='*60}")

    if not weight_path.exists():
        print(f"[ERROR] 权重文件不存在: {weight_path}")
        return None

    # 加载模型
    model = YOLO(str(weight_path))

    # 评估 - 标准设置
    print("\n标准评估 (conf=0.001, iou=0.6)...")
    metrics = model.val(
        data=str(test_yaml),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.6,
        augment=False,
        verbose=True,
    )

    results = {
        "name": name,
        "weight_path": str(weight_path),
        "precision": float(metrics.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(metrics.results_dict.get("metrics/recall(B)", 0)),
        "map50": float(metrics.results_dict.get("metrics/mAP50(B)", 0)),
        "map75": float(metrics.results_dict.get("metrics/mAP75(B)", 0)),
        "map50_95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0)),
    }

    # 每类AP
    per_class_ap = {}
    for i, cls_name in enumerate(CLASSES):
        key = f"metrics/mAP50({cls_name})"
        per_class_ap[cls_name] = float(metrics.results_dict.get(key, 0))
    results["per_class_ap"] = per_class_ap

    # TTA评估
    print("\nTTA评估 (conf=0.001, augment=True)...")
    metrics_tta = model.val(
        data=str(test_yaml),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.6,
        augment=True,
        verbose=False,
    )

    results["precision_tta"] = float(metrics_tta.results_dict.get("metrics/precision(B)", 0))
    results["recall_tta"] = float(metrics_tta.results_dict.get("metrics/recall(B)", 0))
    results["map50_tta"] = float(metrics_tta.results_dict.get("metrics/mAP50(B)", 0))
    results["map50_95_tta"] = float(metrics_tta.results_dict.get("metrics/mAP50-95(B)", 0))

    return results

def print_results(results: dict):
    """打印评估结果"""
    print(f"\n{'='*50}")
    print(f"评估结果: {results['name']}")
    print(f"{'='*50}")

    print("\n总体指标:")
    print(f"  Precision:      {results['precision']:.4f}")
    print(f"  Recall:         {results['recall']:.4f}")
    print(f"  mAP@0.5:        {results['map50']:.4f}")
    print(f"  mAP@0.75:       {results['map75']:.4f}")
    print(f"  mAP@0.5:0.95:   {results['map50_95']:.4f}")

    print("\nTTA (Test Time Augmentation):")
    print(f"  Precision:      {results['precision_tta']:.4f}")
    print(f"  Recall:         {results['recall_tta']:.4f}")
    print(f"  mAP@0.5:        {results['map50_tta']:.4f}")
    print(f"  mAP@0.5:0.95:   {results['map50_95_tta']:.4f}")

    print("\n每类 mAP@0.5:")
    for cls_name, ap in results["per_class_ap"].items():
        bar = "█" * int(ap * 20)
        print(f"  {cls_name:<20} {ap:.4f} {bar}")

def save_results_csv(all_results: list, output_path: Path):
    """保存结果为CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if not all_results:
            return

        # 准备字段
        fieldnames = ["name", "precision", "recall", "map50", "map75", "map50_95",
                     "precision_tta", "recall_tta", "map50_tta", "map50_95_tta"]
        fieldnames += [f"ap_{cls}" for cls in CLASSES]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_results:
            row = {k: r.get(k, "") for k in fieldnames if k in r}
            # 添加每类AP
            for cls_name, ap in r.get("per_class_ap", {}).items():
                row[f"ap_{cls_name}"] = ap
            writer.writerow(row)

    print(f"\nCSV结果已保存: {output_path}")

def save_results_json(all_results: list, output_path: Path):
    """保存结果为JSON"""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"JSON结果已保存: {output_path}")

# ============================================================
# 主流程
# ============================================================

def main():
    print("="*60)
    print("模型评估工具")
    print("="*60)
    print(f"评估模型数量: {len(MODELS_TO_EVALUATE)}")

    # 创建测试yaml
    test_yaml = create_test_yaml()

    # 评估所有模型
    all_results = []
    for name, weight_path in MODELS_TO_EVALUATE:
        results = evaluate_model(
            model=None,
            weight_path=Path(weight_path),
            test_yaml=test_yaml,
            name=name,
        )
        if results:
            print_results(results)
            all_results.append(results)

    # 保存结果
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results_csv(all_results, OUTPUT_ROOT / "evaluation_results.csv")
        save_results_json(all_results, OUTPUT_ROOT / "evaluation_results.json")

        # 打印汇总表
        print("\n" + "="*80)
        print("评估汇总")
        print("="*80)
        print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['name']:<30} {r['precision']:<12.4f} {r['recall']:<12.4f} "
                  f"{r['map50']:<12.4f} {r['map50_95']:<12.4f}")
        print("="*80)

    else:
        print("\n没有成功评估任何模型!")

if __name__ == "__main__":
    main()
