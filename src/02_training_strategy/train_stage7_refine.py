from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    modelpath = ROOT / "experiments/stage6_semi/weights/best-640.pt"
    datapath = ROOT / "datasets/neu.yaml"
    project_path = ROOT / "experiments"

    model = YOLO(str(modelpath))

    model.train(
        data=str(datapath),
        imgsz=640,
        epochs=30,
        patience=15,
        batch=16,
        lr0=1e-4,
        lrf=1e-3,
        mosaic=0.0, # 关闭数据增强，学习真正的图
        flipud=0.0,
        fliplr=0.0,
        box=7.5, # 默认定位权重
        cls=1.2, # 纠正分类噪声

        project=str(project_path),
        name="stage7_refine_compare"
    )

if __name__ == "__main__":
    main()
