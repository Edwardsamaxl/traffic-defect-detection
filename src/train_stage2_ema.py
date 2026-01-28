from ultralytics import YOLO
from pathlib import Path


def main():
    yaml_path = "../ultralytics-main/ultralytics/cfg/models/v8/yolov8n_ema.yaml"
    data_yaml = "../datasets/neu.yaml"

    model = YOLO(yaml_path)

    model.train(
        data=data_yaml,
        epochs=120,
        patience=50,
        imgsz=640,
        batch=4,

        project="E:/PycharmProjects/traffic-defect-detection/experiments",
        name="stage2_ema",
    )

if __name__ == "__main__":
    main()
