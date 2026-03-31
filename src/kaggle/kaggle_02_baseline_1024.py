"""
Kaggle训练脚本 2: 高分辨率模型 (1024分辨率)
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
        imgsz=1024,  # 高分辨率，捕获细粒度缺陷
        batch=2,      # 1024需要更小batch
        project=str(ROOT / "experiments"),
        name="baseline_1024",
    )

if __name__ == "__main__":
    main()
