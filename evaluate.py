"""
增强评估入口

使用示例:
    # 评估单个实验
    python evaluate.py --exp baseline --model experiments/baseline/weights/best.pt

    # 批量评估
    python evaluate.py --batch

    # 详细分析
    python evaluate.py --exp baseline --analyze
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cfg import get_strategy
from src.utils.config import EXPERIMENTS_ROOT, REPORTS_ROOT, DATASETS
from src.utils.evaluator import EnhancedEvaluator, batch_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="评估工具")

    parser.add_argument("--exp", type=str, default=None,
                        help="实验名称")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径")
    parser.add_argument("--data", type=str, default="neu",
                        help="数据集配置")
    parser.add_argument("--split", type=str, default="test",
                        help="评估 split")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="图像尺寸")
    parser.add_argument("--tta", action="store_true",
                        help="使用 TTA")

    # 分析选项
    parser.add_argument("--analyze", action="store_true",
                        help="执行详细分析（混淆矩阵、错误案例）")

    # 批量评估
    parser.add_argument("--batch", action="store_true",
                        help="批量评估所有预定义实验")

    return parser.parse_args()


def evaluate_single(
    exp_name: str,
    model_path: str | Path,
    data: str = "neu",
    **kwargs
):
    """评估单个实验"""
    # 解析数据路径
    data_yaml = DATASETS.get(data, Path(data))
    if not isinstance(data_yaml, Path):
        data_yaml = Path(data_yaml)

    evaluator = EnhancedEvaluator(
        experiment_name=exp_name,
        model_path=model_path,
        data_yaml=data_yaml,
        split=kwargs.get("split", "test"),
        conf=kwargs.get("conf", 0.001),
        iou=kwargs.get("iou", 0.6),
        imgsz=kwargs.get("imgsz", 640),
    )

    # 评估
    results = evaluator.evaluate(tta=kwargs.get("tta", False))
    evaluator.print_class_summary()

    # 详细分析
    if kwargs.get("analyze", False):
        print("\n执行详细分析...")
        evaluator.analyze_predictions(save_images=True)
        evaluator.compute_confusion_matrix()
        evaluator.plot_confusion_matrix()
        evaluator.save_failure_cases(max_cases=30)

    return results


def evaluate_batch():
    """批量评估预定义实验"""
    experiments = [
        {
            "name": "baseline_s",
            "model_path": EXPERIMENTS_ROOT / "baseline_s" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu"],
        },
        {
            "name": "baseline_s_advanced",
            "model_path": EXPERIMENTS_ROOT / "baseline_s_advanced" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu"],
        },
        {
            "name": "cosine_100",
            "model_path": EXPERIMENTS_ROOT / "cosine_100" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu"],
        },
        {
            "name": "ablation_no_aug",
            "model_path": EXPERIMENTS_ROOT / "ablation_no_aug" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu"],
        },
        {
            "name": "res_640",
            "model_path": EXPERIMENTS_ROOT / "res_640" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu"],
        },
        {
            "name": "res_1024",
            "model_path": EXPERIMENTS_ROOT / "res_1024" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu"],
        },
        {
            "name": "semi_adaptive",
            "model_path": EXPERIMENTS_ROOT / "semi_adaptive" / "weights" / "best.pt",
            "data_yaml": DATASETS["neu_merge"],
        },
    ]

    # 过滤存在的模型
    valid_experiments = []
    for exp in experiments:
        if exp["model_path"].exists():
            valid_experiments.append(exp)
        else:
            print(f"[SKIP] 模型不存在: {exp['model_path']}")

    output_csv = REPORTS_ROOT / "batch_evaluation.csv"
    results = batch_evaluate(valid_experiments, output_csv)

    # 打印对比表格
    print("\n" + "="*80)
    print("实验对比")
    print("="*80)
    print(f"{'实验':<30} {'P':<10} {'R':<10} {'mAP@0.5':<12} {'mAP@0.5:95':<12}")
    print("-"*80)
    for r in results:
        if "error" in r:
            print(f"{r['experiment']:<30} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['experiment']:<30} {r['precision']:<10.4f} {r['recall']:<10.4f} "
                  f"{r['map50']:<12.4f} {r['map50_95']:<12.4f}")
    print("="*80 + "\n")

    return results


def main():
    args = parse_args()

    if args.batch:
        evaluate_batch()
        return

    if not args.exp:
        print("错误: 请指定 --exp 实验名称")
        sys.exit(1)

    if not args.model:
        print("错误: 请指定 --model 模型路径")
        sys.exit(1)

    evaluate_single(
        exp_name=args.exp,
        model_path=args.model,
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tta=args.tta,
        analyze=args.analyze,
    )


if __name__ == "__main__":
    main()
