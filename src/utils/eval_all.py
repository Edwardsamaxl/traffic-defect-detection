"""
批量评估所有实验模型
"""
import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
NEU_DATA = ROOT / "datasets/neu.yaml"

# 所有需要评估的实验
EXPERIMENTS = {
    "baseline": "experiments/baseline/weights/best.pt",
    "baseline_seed": "experiments/baseline_seed/weights/best-conservative.pt",
    "02_cbam": "experiments/02_cbam/weights/best.pt",
    "03_p2_layer": "experiments/exp03_p2_layer/weights/best.pt",
    "04_combined_cbam_p2": "experiments/exp04_combined_cbam_p2_640/weights/best.pt",
    "06a_wiou": "experiments/exp06a_wiou3/weights/best.pt",
}

def evaluate_model(name, weight_path, imgsz=640):
    """评估单个模型"""
    full_path = ROOT / weight_path
    if not full_path.exists():
        print(f"[SKIP] {name}: {full_path} not found")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Weight: {full_path}")
    print(f"{'='*60}")

    model = YOLO(str(full_path))
    metrics = model.val(
        data=str(NEU_DATA),
        imgsz=imgsz,
        conf=0.001,
        iou=0.6,
        split="test",
        augment=False,
        verbose=True,
    )

    results = metrics.results_dict
    results["model_name"] = name
    return results

def main():
    all_results = []

    for name, weight_path in EXPERIMENTS.items():
        result = evaluate_model(name, weight_path)
        if result:
            all_results.append(result)

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'P':<8} {'R':<8} {'mAP50':<8} {'mAP50-95':<10}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['model_name']:<25} "
              f"{r['metrics/precision(B)']:<8.4f} "
              f"{r['metrics/recall(B)']:<8.4f} "
              f"{r['metrics/mAP50(B)']:<8.4f} "
              f"{r['metrics/mAP50-95(B)']:<10.4f}")

    # 保存JSON
    output_path = ROOT / "experiments" / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
