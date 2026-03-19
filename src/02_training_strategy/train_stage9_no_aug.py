"""无增强消融：关闭 mosaic / flipud / fliplr，用于对比有增强的 stage4。"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]  # Kaggle: 项目根为 /kaggle/working/traffic-defect-detection


def main():
    model_path = ROOT / "yolov8s.pt"
    data_yaml = ROOT / "datasets/neu.yaml"
    project_dir = ROOT / "experiments"

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=200,
        patience=50,
        # 消融：关闭增强
        mosaic=0.0,
        flipud=0.0,   # 正确拼写是 flipud（flip up-down）
        fliplr=0.0,    # 正确拼写是 fliplr（flip left-right）
        project=str(project_dir),
        name="stage9_no_aug",
    )


if __name__ == "__main__":
    main()
