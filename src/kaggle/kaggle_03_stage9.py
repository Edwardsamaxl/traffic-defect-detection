"""
Kaggle训练脚本 3: Stage9 - 无增强消融实验
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
        patience=50,
        # 无增强消融
        mosaic=0.0,
        flipud=0.0,
        fliplr=0.0,
        project=str(ROOT / "experiments"),
        name="stage9_no_aug",
    )

if __name__ == "__main__":
    main()
