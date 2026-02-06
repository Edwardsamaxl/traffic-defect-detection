from ultralytics import YOLO

def main():

    model = YOLO("../experiments/stage6_semi/weights/best-640.pt")

    model.train(
        data="../datasets/neu.yaml",
        imgsz=640,
        epochs=30,
        patience=15,
        batch=16,
        lr0=1e-4,
        lrf=1e-3,
        mosaic=0.0,
        flipud=0.0,
        fliplr=0.0,

        box=7.5,  # 稍微提高定位权重
        cls=1.2,  # 纠正分类噪声

        project="../experiments",
        name="stage7_refine_compare"
    )


if __name__ == "__main__":
    main()
