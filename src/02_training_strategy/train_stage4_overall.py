from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def main():
    modelpath = ROOT / "yolov8s.pt"
    datapath = ROOT / "datasets/neu.yaml"
    project_path = ROOT / "experiments"

    model = YOLO(str(modelpath))

    model.train(
        data=str(datapath),

        imgsz=640,  # 关键：提升分辨率抓细节
        epochs=200,  # 确保收敛
        mosaic=1.0,  # 保持数据多样性，默认值
        flipud=0.5,  # 开启垂直翻转，默认值
        close_mosaic=20,
        patience=50,

        project=str(project_path),
        name="stage4_overall",
    )

if __name__ == "__main__":
    main()
