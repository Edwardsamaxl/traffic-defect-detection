from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="../datasets/neu.yaml",
        epochs=120,
        patience=50,
        imgsz=640,
        batch=4,
        project="E:/PycharmProjects/traffic-defect-detection/experiments",
        name="baseline"
    )

if __name__ == "__main__":
    main()
