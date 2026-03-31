"""
Kaggle训练脚本 4: Copy-Paste增强 (1024分辨率)
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")

def create_yaml():
    yaml_path = ROOT / "datasets/neu_copy_paste.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {ROOT}/data/NEU-DET/train_copy_paste
train: images/train
val: ../images/val
test: ../images/test

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
    yaml_path = ROOT / "datasets/neu_copy_paste.yaml"
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(yaml_path),
        epochs=200,
        patience=50,
        imgsz=1024,
        batch=2,
        project=str(ROOT / "experiments"),
        name="copypaste_1024",
    )

if __name__ == "__main__":
    main()
