"""
Kaggle训练脚本 4: Copy-Paste增强 (1024分辨率)
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
        imgsz=1024,
        batch=2,
        project=str(ROOT / "experiments"),
        name="copypaste_1024",
    )

if __name__ == "__main__":
    main()
