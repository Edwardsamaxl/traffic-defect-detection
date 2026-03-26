"""
快速开始脚本 - 新手友好

使用方法:
    python quick_start.py                    # 显示帮助
    python quick_start.py info              # 显示系统信息
    python quick_start.py train baseline    # 训练 baseline
    python quick_start.py eval baseline     # 评估 baseline
    python quick_start.py list              # 列出所有策略
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_device, get_device_name, get_memory_info, CLASSES
from src.cfg import list_strategies, get_strategy


def print_info():
    """打印系统信息"""
    print("\n" + "="*60)
    print("系统信息")
    print("="*60)
    print(f"设备: {get_device_name()}")
    print(f"计算设备: {get_device()}")

    mem = get_memory_info()
    if mem["total_gb"] > 0:
        print(f"显存: {mem['total_gb']:.1f} GB")
        print(f"  已用: {mem['allocated_gb']:.2f} GB")
        print(f"  空闲: {mem['free_gb']:.2f} GB")
    else:
        print("显存: N/A (使用 CPU)")

    print("\n缺陷类别:")
    for i, cls in enumerate(CLASSES):
        print(f"  {i}: {cls}")

    print("="*60 + "\n")


def print_strategies():
    """列出所有策略"""
    strategies = list_strategies()

    print("\n可用训练策略:")
    print("-" * 50)

    # 分类显示
    supervised = [s for s in strategies if "baseline" in s or "res_" in s or "cosine" in s]
    augmentation = [s for s in strategies if "aug" in s or "copy_paste" in s]
    semi = [s for s in strategies if "seed" in s or "semi" in s or "curriculum" in s]

    print("\n[监督学习]")
    for s in sorted(supervised):
        cfg = get_strategy(s)
        print(f"  {s:<30} epochs={cfg.epochs}, batch={cfg.batch}, cos_lr={cfg.cos_lr}")

    print("\n[数据增强]")
    for s in sorted(augmentation):
        cfg = get_strategy(s)
        print(f"  {s:<30} mosaic={cfg.mosaic}, mixup={cfg.mixup}")

    print("\n[半监督学习]")
    for s in sorted(semi):
        cfg = get_strategy(s)
        print(f"  {s:<30} data={cfg.data}")

    print("\n" + "-" * 50)
    print(f"共 {len(strategies)} 个策略\n")


def quick_train(strategy: str):
    """快速训练"""
    from src.utils.trainer import ExperimentRunner

    cfg = get_strategy(strategy)
    if cfg is None:
        print(f"错误: 未知策略 '{strategy}'")
        print("使用 --list 查看所有策略")
        return

    print(f"\n开始训练: {strategy}")
    print(f"模型: {cfg.model}")
    print(f"数据: {cfg.data}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch: {cfg.batch}")
    print()

    runner = ExperimentRunner(strategy, cfg)
    result = runner.train()

    print("\n训练完成，开始评估...")
    runner.eval(conf=0.001, iou=0.6)


def quick_eval(strategy: str):
    """快速评估"""
    from src.utils.evaluator import EnhancedEvaluator
    from src.utils.config import EXPERIMENTS_ROOT, DATASETS

    cfg = get_strategy(strategy)
    if cfg is None:
        print(f"错误: 未知策略 '{strategy}'")
        return

    # 查找模型
    model_path = EXPERIMENTS_ROOT / strategy / "weights" / "best.pt"
    if not model_path.exists():
        for alt in ["new-best.pt", "best-cosine.pt", "last.pt"]:
            alt_path = EXPERIMENTS_ROOT / strategy / "weights" / alt
            if alt_path.exists():
                model_path = alt_path
                break

    if not model_path.exists():
        print(f"错误: 未找到模型 {model_path}")
        print("请先训练模型: python quick_start.py train " + strategy)
        return

    data_yaml = DATASETS.get(cfg.data, Path(cfg.data))

    evaluator = EnhancedEvaluator(
        experiment_name=strategy,
        model_path=model_path,
        data_yaml=data_yaml,
    )

    results = evaluator.evaluate(tta=False)
    evaluator.print_class_summary()


def main():
    parser = argparse.ArgumentParser(description="快速开始")
    parser.add_argument("action", nargs="?", choices=["info", "list", "train", "eval"],
                        help="操作: info=系统信息, list=策略列表, train=训练, eval=评估")
    parser.add_argument("strategy", nargs="?", help="策略名称")

    args = parser.parse_args()

    if args.action == "info":
        print_info()
    elif args.action == "list":
        print_strategies()
    elif args.action == "train":
        if not args.strategy:
            print("错误: 请指定策略名称")
            print("使用 python quick_start.py list 查看所有策略")
            return
        quick_train(args.strategy)
    elif args.action == "eval":
        if not args.strategy:
            print("错误: 请指定策略名称")
            return
        quick_eval(args.strategy)
    else:
        # 默认显示帮助
        print("快速开始脚本")
        print("="*60)
        print("用法:")
        print("  python quick_start.py info              # 显示系统信息")
        print("  python quick_start.py list             # 列出所有策略")
        print("  python quick_start.py train <策略>      # 训练模型")
        print("  python quick_start.py eval <策略>       # 评估模型")
        print()
        print("示例:")
        print("  python quick_start.py train baseline_s_advanced")
        print("  python quick_start.py eval baseline_s_advanced")
        print("="*60)


if __name__ == "__main__":
    main()
