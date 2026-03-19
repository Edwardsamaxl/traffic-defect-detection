from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a YOLO model on NEU-DET splits.")
    p.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "experiments/stage4_overall/weights/best-cosine.pt"),
        help="Path to .pt weights (absolute or relative to project root).",
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "datasets/neu.yaml"),
        help="Dataset yaml path (absolute or relative to project root).",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    p.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for evaluation.")
    p.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS during evaluation.")
    p.add_argument("--tta", action="store_true", help="Enable TTA (augment=True).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_yaml = Path(args.data)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    if not data_yaml.is_absolute():
        data_yaml = (ROOT / data_yaml).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        augment=bool(args.tta),
        verbose=False,
    )

    results = metrics.results_dict
    print("\n===== Config =====")
    print(f"model : {model_path}")
    print(f"data  : {data_yaml}")
    print(f"split : {args.split}")
    print(f"imgsz : {args.imgsz}")
    print(f"tta   : {bool(args.tta)}")

    print("\n===== Overall Metrics =====")
    print(f"Precision      : {results['metrics/precision(B)']:.4f}")
    print(f"Recall         : {results['metrics/recall(B)']:.4f}")
    print(f"mAP@0.5        : {results['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95   : {results['metrics/mAP50-95(B)']:.4f}")


if __name__ == "__main__":
    main()
