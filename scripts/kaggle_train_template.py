"""
Kaggle 训练脚本模板

复制此文件到 Kaggle Notebook 中使用
"""
import sys
sys.path.insert(0, "/kaggle/working/traffic-defect-detection")

from src.cfg import get_strategy
from src.utils.trainer import ExperimentRunner
from src.utils.wandb_integration import ExperimentTracker
from src.utils.config import EXPERIMENTS_ROOT, ensure_dirs

# ========== 配置 ==========
STRATEGY_NAME = "baseline_s_advanced"  # 替换为你的策略
USE_WANDB = True  # 是否使用 Wandb

# ========== 主训练函数 ==========
def train_on_kaggle():
    # 获取策略配置
    cfg = get_strategy(STRATEGY_NAME)
    if cfg is None:
        raise ValueError(f"Unknown strategy: {STRATEGY_NAME}")

    print(f"\n{'='*60}")
    print(f"Training Strategy: {STRATEGY_NAME}")
    print(f"{'='*60}")
    print(f"Config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # 初始化 Wandb
    tracker = None
    if USE_WANDB:
        try:
            tracker = ExperimentTracker(
                experiment_name=STRATEGY_NAME,
                config=cfg.to_dict(),
                use_wandb=True,
            )
            print("Wandb initialized")
        except Exception as e:
            print(f"Wandb init failed: {e}")

    # 创建实验目录
    ensure_dirs(EXPERIMENTS_ROOT)

    # 创建执行器
    runner = ExperimentRunner(
        strategy_name=STRATEGY_NAME,
        from_cfg=cfg,
        project=EXPERIMENTS_ROOT,
    )

    # 训练
    result = runner.train()

    # 评估
    print("\nEvaluating...")
    eval_results = runner.eval(conf=0.001, iou=0.6)

    # 保存摘要
    if tracker:
        tracker.log_metrics({
            "final_precision": eval_results.get("metrics/precision(B)", 0),
            "final_recall": eval_results.get("metrics/recall(B)", 0),
            "final_map50": eval_results.get("metrics/mAP50(B)", 0),
            "final_map50_95": eval_results.get("metrics/mAP50-95(B)", 0),
        })
        tracker.save_summary({
            "best_map50": eval_results.get("metrics/mAP50(B)", 0),
        })
        tracker.finish()

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Results:")
    print(f"  mAP@0.5: {eval_results.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:95: {eval_results.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"{'='*60}\n")

    return result, eval_results


# ========== 批量训练函数 ==========
def train_multiple_strategies(*strategy_names):
    """批量训练多个策略"""
    results = []

    for name in strategy_names:
        cfg = get_strategy(name)
        if cfg is None:
            print(f"[SKIP] Unknown strategy: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}\n")

        runner = ExperimentRunner(name, cfg, project=EXPERIMENTS_ROOT)
        result = runner.train()
        eval_results = runner.eval(conf=0.001, iou=0.6)

        results.append({
            "strategy": name,
            "map50": eval_results.get("metrics/mAP50(B)", 0),
            "map50_95": eval_results.get("metrics/mAP50-95(B)", 0),
        })

    # 打印对比表格
    print("\n" + "="*80)
    print("Strategy Comparison")
    print("="*80)
    print(f"{'Strategy':<30} {'mAP@0.5':<15} {'mAP@0.5:95':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['strategy']:<30} {r['map50']:<15.4f} {r['map50_95']:<15.4f}")
    print("="*80 + "\n")

    return results


# ========== 半监督训练函数 ==========
def train_semi_supervised():
    """半监督训练流程"""
    from src.semi import PseudoLabelGenerator, FixMatchTrainer

    # 步骤 1: 用 seed 数据训练教师模型
    print("\n[Step 1] Training teacher model on seed data...")
    teacher_cfg = get_strategy("seed_supervised")
    teacher_runner = ExperimentRunner("seed_supervised", teacher_cfg)
    teacher_result = teacher_runner.train()
    teacher_weight = teacher_runner.best_weight

    # 步骤 2: 生成伪标签
    print("\n[Step 2] Generating pseudo labels...")
    generator = PseudoLabelGenerator(
        model_path=teacher_weight,
        data_yaml="datasets/neu.yaml",
    )
    generator.generate_adaptive_consistency(
        unlabeled_dir="data/NEU-DET/unlabeled/images/train",
        output_dir="data/NEU-DET/pseudo_labels/train",
    )

    # 步骤 3: 半监督训练
    print("\n[Step 3] Training with pseudo labels...")
    semi_cfg = get_strategy("semi_adaptive")
    semi_runner = ExperimentRunner("semi_adaptive", semi_cfg)
    semi_result = semi_runner.train()
    semi_eval = semi_runner.eval(conf=0.001, iou=0.6)

    print(f"\nSemi-supervised Results:")
    print(f"  mAP@0.5: {semi_eval.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:95: {semi_eval.get('metrics/mAP50-95(B)', 0):.4f}")


if __name__ == "__main__":
    # 训练单个策略
    train_on_kaggle()

    # 或批量训练:
    # train_multiple_strategies("baseline", "cosine_100", "heavy_aug")

    # 或半监督训练:
    # train_semi_supervised()
