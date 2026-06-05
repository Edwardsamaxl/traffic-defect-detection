"""
模型评测工具
=====================================
支持在测试集上评估YOLO模型的性能

用法:
    直接在 PyCharm 里点击运行即可，参数在下方顶部配置区修改
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/neu.yaml"

# ==================== 评测配置（修改这里即可） ====================
DEFAULT_MODEL = ROOT / "experiments/thesis_model/weights/baseline.pt"  # 模型路径
DEFAULT_SPLIT = "test"   # 数据划分: "train" / "val" / "test"
DEFAULT_CONF = 0.001     # 置信度阈值
DEFAULT_IOU = 0.6        # IoU阈值
DEFAULT_AUGMENT = False  # 是否开启TTA（True/False）
# =============================================================


def evaluate(model_path: str = None, data_yaml: str = None, split: str = DEFAULT_SPLIT,
             imgsz: int = 640, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
             augment: bool = DEFAULT_AUGMENT, verbose: bool = True):
    """评估模型性能

    Args:
        model_path: 模型权重路径 (默认: DEFAULT_MODEL)
        data_yaml: 数据集配置文件路径
        split: 数据集划分 ('train', 'val', 'test')
        imgsz: 输入图像尺寸
        conf: 置信度阈值
        iou: IoU阈值
        augment: 是否开启TTA
        verbose: 是否打印详细信息

    Returns:
        results: 评测结果字典
    """
    if data_yaml is None:
        data_yaml = str(DEFAULT_DATA)

    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    model = YOLO(model_path)

    if verbose:
        print(f"\n{'='*50}")
        print(f"模型: {model_path}")
        print(f"数据集: {data_yaml}")
        print(f"划分: {split}")
        print(f"TTA: {'开启' if augment else '关闭'}")
        print(f"{'='*50}\n")

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        split=split,
        augment=augment,
        verbose=verbose,
    )

    return metrics.results_dict


def print_results(results: dict):
    """格式化打印评测结果"""
    print("\n" + "=" * 50)
    print("Overall Metrics")
    print("=" * 50)
    print(f"Precision      : {results['metrics/precision(B)']:.4f}")
    print(f"Recall         : {results['metrics/recall(B)']:.4f}")
    print(f"mAP@0.5        : {results['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95  : {results['metrics/mAP50-95(B)']:.4f}")

    # Per-class metrics
    print("\n" + "=" * 50)
    print("Per-class mAP@0.5")
    print("=" * 50)
    for k, v in results.items():
        if "metrics/mAP50(" in k and k.endswith(")"):
            class_name = k.split("(")[1].rstrip(")")
            print(f"  {class_name:20s}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="YOLO模型评测工具")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help=f"模型权重路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集yaml路径 (默认: datasets/neu.yaml)")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        choices=["train", "val", "test"],
                        help=f"数据集划分 (默认: {DEFAULT_SPLIT})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"置信度阈值 (默认: {DEFAULT_CONF})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help=f"IoU阈值 (默认: {DEFAULT_IOU})")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式，仅打印结果")

    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        augment=DEFAULT_AUGMENT,
        verbose=not args.quiet,
    )

    if args.quiet:
        print_results(results)


if __name__ == "__main__":
    main()
