"""
Kaggle训练脚本 7: 半监督学习 (三种策略)

用法:
    # 训练所有三种策略
    python kaggle_07_semi_supervised.py --strategy all

    # 训练单个策略
    python kaggle_07_semi_supervised.py --strategy standard
    python kaggle_07_semi_supervised.py --strategy adaptive
    python kaggle_07_semi_supervised.py --strategy adaptive_consistency
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/kaggle/working/traffic-defect-detection")


def create_yaml(strategy: str, data_root: Path) -> Path:
    """动态创建数据集yaml配置"""
    merge_dir_map = {
        "standard": "merge-standard",
        "adaptive": "merge-adaptive",
        "adaptive_consistency": "merge-adaptive_consistency",
    }
    yaml_name_map = {
        "standard": "neu_merge_standard",
        "adaptive": "neu_merge_adaptive",
        "adaptive_consistency": "neu_merge_adaptive_consistency",
    }

    yaml_name = yaml_name_map[strategy]
    yaml_path = ROOT / f"datasets/{yaml_name}.yaml"

    with open(yaml_path, "w") as f:
        f.write(f"""path: {data_root}
train: {merge_dir_map[strategy]}/images/train
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


def train_semi(strategy: str):
    """训练单个半监督策略"""
    data_root = ROOT / "data/NEU-DET"
    yaml_path = create_yaml(strategy, data_root)

    project_name_map = {
        "standard": "semi_standard",
        "adaptive": "semi_adaptive_new",
        "adaptive_consistency": "semi_adaptive_consistency_new",
    }

    print(f"\n{'='*60}")
    print(f"训练策略: {strategy}")
    print(f"数据配置: {yaml_path}")
    print(f"{'='*60}")

    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.train(
        data=str(yaml_path),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        project=str(ROOT / "experiments"),
        name=project_name_map[strategy],
    )

    # 评测
    print(f"\n评测 {strategy}...")
    results = model.val(data=str(yaml_path), split="test", conf=0.001, iou=0.6)
    print(f"{strategy} mAP50: {results.box.map50:.4f}")
    print(f"{strategy} mAP50-95: {results.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="半监督训练")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "standard", "adaptive", "adaptive_consistency"],
                        help="训练策略 (默认: all)")
    args = parser.parse_args()

    strategies = ["standard", "adaptive", "adaptive_consistency"] if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        print(f"\n{'#'*60}")
        print(f"# 开始训练: {strategy}")
        print(f"{'#'*60}")
        train_semi(strategy)


if __name__ == "__main__":
    main()
