from ultralytics import YOLO

def main():
    # 使用 Stage2 或 Stage1 的 best.pt
    model = YOLO("../experiments/baseline_s/weights/best.pt")

    model.train(
        data="../datasets/neu.yaml",
        epochs=25,
        patience=10,
        imgsz=640,
        batch=4,

        lr0=1e-4,
        lrf=1e-3,

        freeze=10,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        optimizer="SGD",
        momentum=0.937,
        weight_decay=5e-4,

        amp=False,
        project="../experiments",
        name="stage3_refine"
    )


if __name__ == "__main__":
    main()
