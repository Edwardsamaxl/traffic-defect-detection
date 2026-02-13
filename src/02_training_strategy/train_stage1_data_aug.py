from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    modelpath = ROOT / "src/yolov8n.pt"
    datapath = ROOT / "datasets/neu.yaml"
    project_path = ROOT / "experiments"

    model = YOLO(str(modelpath))

    model.train(
        data=str(datapath),

        epochs=120,
        patience=50,
        imgsz=640,
        batch=4,

        # ====== Stage 1：数据增强 ======
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,

        degrees=3.0,
        translate=0.03,
        scale=0.15,

        project=str(project_path),
        name="stage1_data_aug",
    )

if __name__ == "__main__":
    main()
