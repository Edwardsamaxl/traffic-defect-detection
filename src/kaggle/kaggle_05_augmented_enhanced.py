"""
Kaggle训练脚本 5: 增强数据增强 (640分辨率)
Exp-05: 通过更激进的数据增强策略提升模型鲁棒性

增强策略: mosaic, mixup, hsv, degrees, translate, scale, shear, perspective, erasing
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")

def create_yaml():
    yaml_path = ROOT / "datasets/neu_augmented.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {ROOT}/data/NEU-DET
train: images/train
val: images/val
test: images/test

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
    yaml_path = ROOT / "datasets/neu_augmented.yaml"
    model = YOLO(str(ROOT / "yolov8s.pt"))

    model.train(
        data=str(yaml_path),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        project=str(ROOT / "experiments"),
        name="exp05_augment_enhanced",
        # 增强数据增强配置
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        erasing=0.4,
        fliplr=0.5,
    )

if __name__ == "__main__":
    main()
