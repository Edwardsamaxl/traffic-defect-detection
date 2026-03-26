"""
批量运行所有实验的脚本

用法:
    python scripts/run_all_experiments.py --phase train    # 运行所有训练
    python scripts/run_all_experiments.py --phase eval    # 运行所有评估
    python scripts/run_all_experiments.py --phase all     # 训练 + 评估
"""
import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cfg import list_strategies, get_strategy
from src.utils.config import EXPERIMENTS_ROOT, REPORTS_ROOT, DATASETS, ensure_dirs
from src.utils.trainer import ExperimentRunner
from src.utils.evaluator import batch_evaluate


# 定义要运行的实验
SUPERVISED_EXPERIMENTS = [
    "baseline_s_advanced",
    "ablation_no_aug",
    "res_640",
    "res_1024",
    "cosine_100",
    "heavy_aug",
    "light_aug",
    "copy_paste",
]

SEMI_SUPERVISED_EXPERIMENTS = [
    "seed_supervised",
    "seed_supervised_advanced",
    "semi_adaptive",
    "semi_adaptive_conservative",
]

ABLATION_EXPERIMENTS = [
    "ablation_no_aug_scratch",
    "cosine_30",
]


def run_training(experiments: list[str], project: Path) -> list[dict]:
    """运行训练"""
    results = []

    for name in experiments:
        cfg = get_strategy(name)
        if cfg is None:
            print(f"[SKIP] Unknown strategy: {name}")
            continue

        print(f"\n{'#'*60}")
        print(f"# Training: {name}")
        print(f"{'#'*60}\n")

        try:
            runner = ExperimentRunner(name, cfg, project=project)
            result = runner.train()
            results.append({"name": name, "status": "success", "result": result})
        except Exception as e:
            print(f"[ERROR] Training {name} failed: {e}")
            results.append({"name": name, "status": "failed", "error": str(e)})

    return results


def run_evaluation(experiments: list[dict], output_csv: Path):
    """运行评估"""
    # 转换为评估器格式
    eval_configs = []
    for exp in experiments:
        if exp["status"] != "success":
            continue

        model_path = EXPERIMENTS_ROOT / exp["name"] / "weights" / "best.pt"
        if not model_path.exists():
            # 尝试其他命名
            for alt_name in ["new-best.pt", "best-cosine.pt", "last.pt"]:
                alt_path = EXPERIMENTS_ROOT / exp["name"] / "weights" / alt_name
                if alt_path.exists():
                    model_path = alt_path
                    break

        if model_path.exists():
            eval_configs.append({
                "name": exp["name"],
                "model_path": model_path,
                "data_yaml": DATASETS["neu"],
            })

    return batch_evaluate(eval_configs, output_csv)


def print_summary(results: list[dict], title: str = "Results"):
    """打印结果摘要"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print(f"\nTotal: {len(results)}, Success: {len(success)}, Failed: {len(failed)}")

    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="批量运行实验")
    parser.add_argument("--phase", type=str, choices=["train", "eval", "all"], default="all",
                        help="运行阶段: train=仅训练, eval=仅评估, all=训练+评估")
    parser.add_argument("--supervised", action="store_true",
                        help="运行监督学习实验")
    parser.add_argument("--semi", action="store_true",
                        help="运行半监督学习实验")
    parser.add_argument("--ablation", action="store_true",
                        help="运行消融实验")
    parser.add_argument("--custom", type=str, nargs="+", default=None,
                        help="自定义实验列表")
    parser.add_argument("--project", type=str, default=None,
                        help="实验输出目录")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的实验")

    args = parser.parse_args()

    # 确定实验列表
    experiments = []
    if args.custom:
        experiments.extend(args.custom)
    else:
        if args.supervised:
            experiments.extend(SUPERVISED_EXPERIMENTS)
        if args.semi:
            experiments.extend(SEMI_SUPERVISED_EXPERIMENTS)
        if args.ablation:
            experiments.extend(ABLATION_EXPERIMENTS)

    if not experiments:
        print("没有指定要运行的实验")
        return

    # 确定项目目录
    project = Path(args.project) if args.project else EXPERIMENTS_ROOT
    ensure_dirs(project)

    print(f"\n{'='*60}")
    print(f"Batch Experiment Runner")
    print(f"{'='*60}")
    print(f"Phase: {args.phase}")
    print(f"Experiments: {len(experiments)}")
    print(f"Project: {project}")
    print(f"{'='*60}\n")

    # 训练
    if args.phase in ["train", "all"]:
        train_results = run_training(experiments, project)
        print_summary(train_results, "Training Summary")

        # 保存训练结果
        train_report = project / "training_results.json"
        with open(train_report, "w") as f:
            json.dump(train_results, f, indent=2)
        print(f"Training results saved: {train_report}")

    # 评估
    if args.phase in ["eval", "all"]:
        # 构建评估配置
        eval_configs = []
        for name in experiments:
            model_path = project / name / "weights" / "best.pt"
            if not model_path.exists():
                for alt in ["new-best.pt", "best-cosine.pt", "last.pt"]:
                    alt_path = project / name / "weights" / alt
                    if alt_path.exists():
                        model_path = alt_path
                        break

            if model_path.exists():
                eval_configs.append({
                    "name": name,
                    "model_path": model_path,
                    "data_yaml": DATASETS["neu"],
                })

        if eval_configs:
            output_csv = REPORTS_ROOT / "all_experiments_evaluation.csv"
            eval_results = batch_evaluate(eval_configs, output_csv)

            # 打印最终对比
            print("\n" + "="*80)
            print("Final Comparison")
            print("="*80)
            print(f"{'Experiment':<30} {'P':<10} {'R':<10} {'mAP@0.5':<12} {'mAP@0.5:95':<12}")
            print("-"*80)
            for r in sorted(eval_results, key=lambda x: x.get("map50", 0), reverse=True):
                if "error" in r:
                    print(f"{r['experiment']:<30} ERROR")
                else:
                    print(f"{r['experiment']:<30} {r['precision']:<10.4f} {r['recall']:<10.4f} "
                          f"{r['map50']:<12.4f} {r['map50_95']:<12.4f}")
            print("="*80 + "\n")

    print("All done!")


if __name__ == "__main__":
    main()
