from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    modelpath = ROOT / "experiments/stage4_overall/weights/best.pt"
    datapath = ROOT / "datasets/neu.yaml"
    project_path = ROOT / "experiments"

    model = YOLO(str(modelpath))

    model.train(
        data=str(datapath),
        epochs=25,
        patience=10,
        imgsz=1024,
        batch=4,

        lr0=1e-4,
        lrf=1e-3,
        mosaic=0.0,

        amp=False,
        project=str(project_path),
        name="stage3_refine_s4"
    )

if __name__ == "__main__":
    main()
