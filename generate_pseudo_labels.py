"""
伪标签生成入口

使用示例:
    # 标准伪标签（固定阈值）
    python generate_pseudo_labels.py --method standard --model experiments/baseline_seed/weights/best.pt

    # 自适应阈值
    python generate_pseudo_labels.py --method adaptive --model experiments/baseline_seed/weights/best.pt

    # 翻转一致性
    python generate_pseudo_labels.py --method consistency --model experiments/baseline_seed/weights/best.pt

    # 自适应 + 一致性组合
    python generate_pseudo_labels.py --method adaptive_consistency --model experiments/baseline_seed/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.semi import PseudoLabelGenerator
from src.utils.config import EXPERIMENTS_ROOT, DATA_ROOT, DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description="伪标签生成")

    parser.add_argument("--method", type=str, required=True,
                        choices=["standard", "adaptive", "consistency", "adaptive_consistency", "uncertainty"],
                        help="伪标签生成方法")
    parser.add_argument("--model", type=str, required=True,
                        help="教师模型路径")
    parser.add_argument("--data", type=str, default="neu",
                        help="数据集配置")
    parser.add_argument("--unlabeled", type=str, required=True,
                        help="无标签图像目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出标签目录")
    parser.add_argument("--conf", type=float, default=0.7,
                        help="固定置信度阈值（standard/consistency 方法）")
    parser.add_argument("--base-conf", type=float, default=0.65,
                        help="自适应阈值基础值")
    parser.add_argument("--lambda", type=float, default=0.25,
                        help="自适应阈值 lambda 参数")
    parser.add_argument("--iou-match", type=float, default=0.6,
                        help="一致性匹配 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="图像尺寸")

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析数据路径
    data_yaml = DATASETS.get(args.data, Path(args.data))
    if not isinstance(data_yaml, Path):
        data_yaml = Path(data_yaml)

    # 创建生成器
    generator = PseudoLabelGenerator(
        model_path=args.model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
    )

    # 生成伪标签
    method = args.method
    unlabeled_dir = Path(args.unlabeled)
    output_dir = Path(args.output)

    print(f"\n{'='*60}")
    print(f"伪标签生成")
    print(f"{'='*60}")
    print(f"方法: {method}")
    print(f"模型: {args.model}")
    print(f"无标签数据: {unlabeled_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")

    if method == "standard":
        generator.generate_standard(
            unlabeled_dir=unlabeled_dir,
            output_dir=output_dir,
            conf_threshold=args.conf,
        )
    elif method == "adaptive":
        generator.generate_adaptive(
            unlabeled_dir=unlabeled_dir,
            output_dir=output_dir,
            base_conf=args.base_conf,
            lambda_val=args.lambda,
        )
    elif method == "consistency":
        generator.generate_consistency(
            unlabeled_dir=unlabeled_dir,
            output_dir=output_dir,
            base_conf=args.conf,
            iou_match=args.iou_match,
        )
    elif method == "adaptive_consistency":
        generator.generate_adaptive_consistency(
            unlabeled_dir=unlabeled_dir,
            output_dir=output_dir,
            base_conf=args.base_conf,
            lambda_val=args.lambda,
            iou_match=args.iou_match,
        )
    elif method == "uncertainty":
        generator.generate_with_uncertainty(
            unlabeled_dir=unlabeled_dir,
            output_dir=output_dir,
            conf_threshold=args.conf,
        )

    print("\n伪标签生成完成!")


if __name__ == "__main__":
    main()
