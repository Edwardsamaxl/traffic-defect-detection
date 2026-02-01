from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="../datasets/neu.yaml",

        imgsz=1024,  # 关键：提升分辨率抓细节
        epochs=200,  # 确保收敛
        fl_gamma=1.5,  # 开启 Focal Loss
        cls=1.0,  # 提高分类权重
        mosaic=1.0,  # 保持数据多样性
        flipud=0.5,  # 开启垂直翻转
        close_mosaic=10,  # 结束前关闭 Mosaic
        patience=50,

        project="../experiments",
        name="stage4_overall",
    )

if __name__ == "__main__":
    main()
