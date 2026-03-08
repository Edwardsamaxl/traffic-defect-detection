from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":

    model_path = ROOT / "experiments/stage4_overall/weights/best-cosine.pt"

    model = YOLO(model_path)
    print(model.model.model)