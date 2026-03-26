"""
数据集分析入口

使用示例:
    # 分析默认数据集
    python analyze_data.py

    # 分析指定数据集
    python analyze_data.py --data data/NEU-DET --name NEU-DET

    # 对比多个数据集
    python analyze_data.py --compare --datasets data/NEU-DET data/NEU-DET-semi --names original semi
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.analysis import DatasetAnalyzer, compare_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="数据集分析")

    parser.add_argument("--data", type=str, default="data/NEU-DET",
                        help="数据集路径")
    parser.add_argument("--name", type=str, default=None,
                        help="数据集名称")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="对比数据集列表")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                        help="对比数据集名称")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        if not args.datasets:
            print("错误: 对比模式需要指定 --datasets")
            sys.exit(1)

        datasets = [Path(d) for d in args.datasets]
        names = args.names if args.names else [d.name for d in datasets]

        print(f"\n对比 {len(datasets)} 个数据集:")
        for d, n in zip(datasets, names):
            print(f"  - {n}: {d}")

        compare_datasets(datasets, names)

    else:
        data_path = Path(args.data)
        name = args.name or data_path.name

        print(f"\n分析数据集: {data_path}")

        analyzer = DatasetAnalyzer(data_path, name)
        report = analyzer.generate_report(save=True)

        print("\n" + "="*60)
        print("数据集统计摘要")
        print("="*60)
        print(f"数据集: {name}")
        print(f"图像总数: {report['num_images']}")
        print(f"标签总数: {report['num_labels']}")
        print(f"目标总数: {report['total_objects']}")
        print(f"Bbox 平均宽度: {report['bbox_stats']['mean_width']:.4f}")
        print(f"Bbox 平均高度: {report['bbox_stats']['mean_height']:.4f}")
        print(f"Bbox 平均宽高比: {report['bbox_stats']['mean_aspect_ratio']:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
