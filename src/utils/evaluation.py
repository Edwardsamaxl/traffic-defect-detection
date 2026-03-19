from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    # ===== 你只需要改这里 =====
    model_path = ROOT / "experiments/stage4_overall/weights/best-cosine.pt"
    data_yaml = ROOT / "datasets/neu.yaml"
    split = "test"  # "val" / "test"
    imgsz = 640
    conf = 0.001
    iou = 0.6
    tta = True
    # =========================

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=tta,
        verbose=False,
    )

    results = metrics.results_dict
    print("\n===== Config =====")
    print(f"model : {model_path}")
    print(f"data  : {data_yaml}")
    print(f"split : {split}")
    print(f"imgsz : {imgsz}")
    print(f"tta   : {tta}")

    print("\n===== Overall Metrics =====")
    print(f"Precision      : {results['metrics/precision(B)']:.4f}")
    print(f"Recall         : {results['metrics/recall(B)']:.4f}")
    print(f"mAP@0.5        : {results['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95   : {results['metrics/mAP50-95(B)']:.4f}")
