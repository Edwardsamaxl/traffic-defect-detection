"""
数据集合并脚本 - 支持三种伪标签策略

将 seed 数据与伪标签数据合并，重新划分为 train/val

用法:
    python src/04_semi_supervised/merge_dataset.py --strategy adaptive_consistency
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/NEU-DET"

# 数据路径配置
SEED_ROOT = DATASET_ROOT / "seed"
UNLABELED_ROOT = DATASET_ROOT / "unlabeled"

# seed: images/train + labels/train (原始有标签数据)
# unlabeled: images/train + pseudo_labels/{strategy}/ (伪标签数据)
# val/test: 使用原始划分的 images/val 和 images/test (仅用于训练)

SEED_RATIO = 0.3  # 从 train 中取 30% 作为 seed
RANDOM_SEED = 42


def get_strategy_paths(strategy):
    pseudo_dir = UNLABELED_ROOT / "pseudo_labels" / strategy / "train"
    return pseudo_dir


def main():
    parser = argparse.ArgumentParser(description="合并半监督数据集")
    parser.add_argument("--strategy", type=str, default="adaptive_consistency",
                        choices=["standard", "adaptive", "adaptive_consistency"],
                        help="伪标签策略")
    parser.add_argument("--seed-ratio", type=float, default=SEED_RATIO,
                        help="从train中采样seed的比例 (默认: 0.3)")
    parser.add_argument("--seed-only", action="store_true",
                        help="只使用seed数据，不合并伪标签")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # 合并输出目录
    merge_root = DATASET_ROOT / f"merge-{args.strategy}"
    pseudo_dir = get_strategy_paths(args.strategy)

    print(f"{'='*60}")
    print(f"合并数据集 - 策略: {args.strategy}")
    print(f"Seed目录: {SEED_ROOT}")
    print(f"Pseudo目录: {pseudo_dir}")
    print(f"合并输出: {merge_root}")
    print(f"{'='*60}")

    # 删除旧目录并重建 (只创建train，val使用原始images/val)
    if merge_root.exists():
        shutil.rmtree(merge_root)
    (merge_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (merge_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # 收集所有训练样本对
    all_pairs = []

    # 1. Seed 数据 (有标签)
    seed_img_dir = SEED_ROOT / "images/train"
    seed_lbl_dir = SEED_ROOT / "labels/train"
    if seed_img_dir.exists():
        for img in seed_img_dir.glob("*"):
            if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue
            lbl = seed_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                all_pairs.append((img, lbl))
        print(f"Seed样本: {len(all_pairs)}")

    # 2. 伪标签数据 (仅在非 seed-only 模式)
    pseudo_img_dir = UNLABELED_ROOT / "images/train"
    pseudo_lbl_dir = pseudo_dir

    if not args.seed_only and pseudo_lbl_dir.exists():
        pseudo_count = 0
        for img in pseudo_img_dir.glob("*"):
            if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue
            lbl = pseudo_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                all_pairs.append((img, lbl))
                pseudo_count += 1
        print(f"Pseudo样本: {pseudo_count}")
    elif args.seed_only:
        print("Seed-only 模式，不合并伪标签")

    print(f"总计样本: {len(all_pairs)}")

    if len(all_pairs) == 0:
        print("[ERROR] 没有找到任何样本！")
        return

    # 打乱
    random.shuffle(all_pairs)

    # 拷贝函数
    for idx, (img, lbl) in enumerate(all_pairs):
        new_name = f"{img.stem}_{idx}{img.suffix}"
        shutil.copy(img, merge_root / "images" / "train" / new_name)
        shutil.copy(lbl, merge_root / "labels" / "train" / f"{Path(new_name).stem}.txt")

    print(f"\n===== 合并完成 =====")
    print(f"Train: {len(all_pairs)}")
    print(f"输出目录: {merge_root}")
    print(f"Val: 使用原始 images/val (有真实标签)")


if __name__ == "__main__":
    main()
