"""
Kaggle训练脚本 2: Stage4 - 标准增强版本
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")

def main():
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(ROOT / "datasets/neu.yaml"),
        imgsz=640,
        epochs=200,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        close_mosaic=20,
        patience=50,
        project=str(ROOT / "experiments"),
        name="stage4_overall",
    )

if __name__ == "__main__":
    main()
