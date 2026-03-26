"""
统一训练入口

使用示例:
    # 使用预定义策略训练
    python train.py --strategy baseline_s_advanced

    # 训练并指定额外参数
    python train.py --strategy baseline --epochs 100 --batch 8

    # 训练多个策略
    python train.py --strategies baseline cosine_100 heavy_aug

    # 仅打印策略配置，不训练
    python train.py --strategy baseline_s_advanced --dry-run
"""
import argparse
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cfg import get_strategy, list_strategies
from src.utils.trainer import ExperimentRunner, train_and_compare
from src.utils.config import EXPERIMENTS_ROOT, ensure_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="统一训练入口")

    parser.add_argument("--strategy", type=str, default=None,
                        help="训练策略名称")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                        help="多个策略名称，依次训练并对比")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖策略的 epochs")
    parser.add_argument("--batch", type=int, default=None,
                        help="覆盖策略的 batch size")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="覆盖策略的图像尺寸")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖策略的设备")
    parser.add_argument("--project", type=str, default=None,
                        help="实验输出目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印配置，不实际训练")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用策略")

    return parser.parse_args()


def main():
    args = parse_args()

    # 列出所有策略
    if args.list:
        print("\n可用训练策略:")
        print("-" * 50)
        for name in list_strategies():
            print(f"  {name}")
        print("-" * 50)
        print(f"共 {len(list_strategies())} 个策略\n")
        return

    # 训练多个策略
    if args.strategies:
        print(f"\n{'='*60}")
        print(f"批量训练 {len(args.strategies)} 个策略")
        print(f"{'='*60}\n")
        results = train_and_compare(*args.strategies)
        return

    # 单策略训练
    if not args.strategy:
        print("错误: 请指定 --strategy 或 --strategies")
        print("使用 --list 查看所有可用策略")
        sys.exit(1)

    # 获取配置
    cfg = get_strategy(args.strategy)
    if cfg is None:
        print(f"错误: 未知的策略 '{args.strategy}'")
        print("使用 --list 查看所有可用策略")
        sys.exit(1)

    # 构建额外参数
    extra_args = {}
    if args.epochs is not None:
        extra_args["epochs"] = args.epochs
    if args.batch is not None:
        extra_args["batch"] = args.batch
    if args.imgsz is not None:
        extra_args["imgsz"] = args.imgsz
    if args.device is not None:
        extra_args["device"] = args.device

    # 确定项目目录
    project = Path(args.project) if args.project else EXPERIMENTS_ROOT
    ensure_dirs(project)

    # 创建执行器
    runner = ExperimentRunner(args.strategy, cfg, project=project, extra_args=extra_args)

    # Dry run
    if args.dry_run:
        runner.print_summary()
        return

    # 训练
    result = runner.train()

    # 自动评估
    print("\n训练完成，开始评估...\n")
    runner.eval(conf=0.001, iou=0.6)

    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)


if __name__ == "__main__":
    main()
