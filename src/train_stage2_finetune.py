from ultralytics import YOLO

def main():
    data_yaml = "../datasets/neu.yaml"

    stage1_weights = (
        "../experiments/stage1_data_aug/weights/best.pt"
    )

    model = YOLO(stage1_weights)

    # ========== Stage 2: Fine-tuning ==========
    model.train(
        data=data_yaml,

        epochs=40,

        # 小学习率微调
        lr0=0.001,
        lrf=0.01,

        optimizer="SGD",

        batch=4,
        imgsz=640,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        patience=20,
        workers=2,
        device=0,
        amp=True,

        project="../experiments",
        name="stage2_finetune",
    )

if __name__ == "__main__":
    main()
