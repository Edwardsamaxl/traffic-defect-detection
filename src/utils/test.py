from ultralytics import YOLO
try:
    from utils.common import PROJECT_ROOT
except ModuleNotFoundError:
    from common import PROJECT_ROOT

ROOT = PROJECT_ROOT

if __name__ == "__main__":

    model_path = ROOT / "experiments/stage4_overall/weights/best-cosine.pt"

    model = YOLO(model_path)
    print(model.model.model)