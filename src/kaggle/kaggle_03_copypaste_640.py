"""
Kaggle训练脚本 3: Copy-Paste增强 (640分辨率)
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")

def main():
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(ROOT / "data/NEU-DET/train_copy_paste/data.yaml"),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        project=str(ROOT / "experiments"),
        name="copypaste_640",
    )

if __name__ == "__main__":
    main()
