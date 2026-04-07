"""
Kaggle模型验证脚本 - 在测试集上评估所有4个Kaggle训练模型
"""
import csv
import os
from pathlib import Path
from ultralytics import YOLO

# 项目根目录
ROOT = Path("E:/PycharmProjects/traffic-defect-detection")
WEIGHTS_DIR = ROOT / "experiments/kaggle/weights"
DATA_YAML = ROOT / "datasets/neu.yaml"
OUTPUT_DIR = ROOT / "experiments/kaggle/validation"

# 4个模型配置: (权重文件, 图像尺寸, 模型简称)
MODELS = [
    ("baseline640.pt", 640, "Baseline-640"),
    ("baseline1024.pt", 1024, "Baseline-1024"),
    ("cp640.pt", 640, "CopyPaste-640"),
    ("cp1024.pt", 1024, "CopyPaste-1024"),
]

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def validate_model(weight_file: str, imgsz: int, model_name: str) -> dict:
    """验证单个模型，返回指标字典"""
    print(f"\n{'='*60}")
    print(f"验证模型: {model_name} (imgsz={imgsz})")
    print(f"{'='*60}")

    weight_path = WEIGHTS_DIR / weight_file
    model = YOLO(str(weight_path))

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        batch=4 if imgsz == 640 else 2,
        verbose=True,
    )

    # 提取指标
    metrics = results.results_dict
    return {
        "model": model_name,
        "weight": weight_file,
        "imgsz": imgsz,
        "mAP50": metrics.get("metrics/mAP50(B)", 0),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
        "precision": metrics.get("metrics/precision(B)", 0),
        "recall": metrics.get("metrics/recall(B)", 0),
        "fitness": metrics.get("fitness", 0),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for weight_file, imgsz, model_name in MODELS:
        weight_path = WEIGHTS_DIR / weight_file
        if not weight_path.exists():
            print(f"⚠️  权重文件不存在: {weight_path}")
            continue

        try:
            result = validate_model(weight_file, imgsz, model_name)
            all_results.append(result)

            # 打印结果
            print(f"\n📊 {model_name} 测试集结果:")
            print(f"   mAP@0.5:   {result['mAP50']:.4f}")
            print(f"   mAP@0.5:0.95: {result['mAP50-95']:.4f}")
            print(f"   Precision: {result['precision']:.4f}")
            print(f"   Recall:   {result['recall']:.4f}")
            print(f"   Fitness:   {result['fitness']:.4f}")

        except Exception as e:
            print(f"❌ 验证失败 {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("没有成功验证任何模型")
        return

    # 保存CSV
    csv_path = OUTPUT_DIR / "validation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n💾 CSV结果已保存: {csv_path}")

    # 生成Markdown表格
    md_path = OUTPUT_DIR / "validation_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Kaggle模型测试集验证结果\n\n")
        f.write("| 模型 | 分辨率 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Fitness |\n")
        f.write("|------|--------|---------|--------------|-----------|--------|--------|\n")
        for r in all_results:
            f.write(f"| {r['model']} | {r['imgsz']} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fitness']:.4f} |\n")
    print(f"💾 Markdown结果已保存: {md_path}")

    # 打印对比表格
    print("\n" + "="*80)
    print("📊 4个模型测试集性能对比")
    print("="*80)
    print(f"{'模型':<18} {'分辨率':>8} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>10} {'Recall':>8}")
    print("-"*80)
    for r in sorted(all_results, key=lambda x: x["mAP50"], reverse=True):
        print(f"{r['model']:<18} {r['imgsz']:>8} {r['mAP50']:>10.4f} {r['mAP50-95']:>14.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}")

if __name__ == "__main__":
    main()