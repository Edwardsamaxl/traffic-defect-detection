from ultralytics import YOLO
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    model_path = ROOT / "experiments/stage7_refine_compare/weights/best.pt"
    #model_path = ROOT / "experiments/baseline_s/weights/best.pt"
    data_yaml = ROOT / "datasets/neu.yaml"

    model = YOLO(model_path)

    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        conf=0.25, # 0.001能够看完整的性能边界，用于测试map
        iou=0.6, # 防止一个物体被框多次，框多次就排除，默认值
    )

    print("\n===== Overall Metrics =====")
    results = metrics.results_dict

    print(f"Precision      : {results['metrics/precision(B)']:.4f}")
    print(f"Recall         : {results['metrics/recall(B)']:.4f}")
    print(f"mAP@0.5        : {results['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95   : {results['metrics/mAP50-95(B)']:.4f}")

    print("\n===== Per-class mAP50 =====")
    for k, v in results.items():
        if "metrics/mAP50(" in k and k.endswith(")"):
            print(f"{k}: {v:.4f}")
