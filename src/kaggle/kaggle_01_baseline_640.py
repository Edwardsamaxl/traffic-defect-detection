"""
Kaggle训练脚本 1: 基线模型 (640分辨率)
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")

def main():
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(ROOT / "datasets/neu.yaml"),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        project=str(ROOT / "experiments"),
        name="baseline_640",
    )

if __name__ == "__main__":
    main()
