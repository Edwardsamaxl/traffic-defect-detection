from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    modelpath = ROOT / "src/yolov8s.pt"
    datapath = ROOT / "datasets/neu_seed.yaml"
    project_path = ROOT / "experiments"

    model = YOLO(str(modelpath))

    model.train(
        data=str(datapath),
        epochs=120,
        patience=50,
        imgsz=640,
        batch=4,
        project=str(project_path),
        name="baseline_seed"
    )

if __name__ == "__main__":
    main()
