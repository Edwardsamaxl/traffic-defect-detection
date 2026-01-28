from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="../datasets/neu.yaml",

        epochs=120,
        patience=50,
        imgsz=640,
        batch=4,

        # ====== Stage 1：数据增强 ======
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        # —— 亮度 / 对比度（近似 gamma / illumination） ——
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,

        # —— 仿射变换（小幅，工业合理） ——
        degrees=3.0,
        translate=0.03,
        scale=0.15,

        project="E:/PycharmProjects/traffic-defect-detection/experiments",
        name="stage1_data_aug",
    )

if __name__ == "__main__":
    main()
