"""
Kaggle训练脚本 6: 基线模型 (seed数据集)
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")

def create_yaml():
    yaml_path = ROOT / "datasets/neu_seed.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {ROOT}/data/NEU-DET-semi/seed
train: images/train
val: images/val
test: ../test/images

names:
 0: crazing
 1: inclusion
 2: patches
 3: pitted_surface
 4: rolled-in_scale
 5: scratches
""")
    return yaml_path

def main():
    create_yaml()
    yaml_path = ROOT / "datasets/neu_seed.yaml"
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(yaml_path),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        project=str(ROOT / "experiments"),
        name="baseline_seed",
    )

if __name__ == "__main__":
    main()
