from ultralytics import YOLO

def main():
    model = YOLO("../yolov8s.pt")

    model.train(
        data="../datasets/neu.yaml",
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        workers=2,
        project="E:/PycharmProjects/traffic-defect-detection/experiments",
        name="baseline_s"
    )

if __name__ == "__main__":
    main()
