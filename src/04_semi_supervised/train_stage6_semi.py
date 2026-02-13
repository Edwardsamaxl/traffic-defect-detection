from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    modelpath = ROOT / "src/yolov8s.pt"
    datapath = ROOT / "datasets/neu_merge.yaml"
    project_path = ROOT / "experiments"

    model = YOLO(str(modelpath))

    model.train(
        data=str(datapath),

        imgsz=640,
        epochs=200,
        fl_gamma=1.5,
        cls=1.0,
        mosaic=1.0,
        flipud=0.5,
        close_mosaic=10,
        patience=50,

        project=str(project_path),
        name="stage6_semi",
    )

if __name__ == "__main__":
    main()
