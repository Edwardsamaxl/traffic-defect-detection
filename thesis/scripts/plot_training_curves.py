"""
绘制训练曲线脚本 - 从训练日志中提取数据并绘制
"""
import json
import csv
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "thesis" / "figures" / "training_curves"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def parse_results_csv(csv_path: Path) -> dict[str, list]:
    """解析Ultralytics的results.csv文件"""
    if not csv_path.exists():
        return {}

    data = {
        'epoch': [],
        'train/box_loss': [],
        'train/cls_loss': [],
        'train/dfl_loss': [],
        'metrics/precision': [],
        'metrics/recall': [],
        'metrics/mAP50': [],
        'metrics/mAP50-95': [],
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data.keys():
                if key in row and row[key]:
                    try:
                        data[key].append(float(row[key]))
                    except ValueError:
                        pass

    return data


def parse_results_json(json_path: Path) -> dict[str, list]:
    """解析results.json文件"""
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        data = json.load(f)

    return data


def plot_training_curves(
    data: dict[str, list],
    title: str,
    output_path: Path,
    smooth: bool = True,
    window: int = 5
):
    """绘制训练曲线"""
    if not data:
        print(f"No data found for {title}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 平滑函数
    def smooth_curve(values: list, window: int) -> list:
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode='valid').tolist()

    epochs = data.get('epoch', list(range(len(data.get('metrics/mAP50', [])))))

    # 1. Loss曲线
    ax = axes[0, 0]
    if 'train/box_loss' in data and data['train/box_loss']:
        losses = data['train/box_loss']
        plot_data = smooth_curve(losses, window) if smooth else losses
        ax.plot(plot_data, label='Box Loss', linewidth=2)
    if 'train/cls_loss' in data and data['train/cls_loss']:
        losses = data['train/cls_loss']
        plot_data = smooth_curve(losses, window) if smooth else losses
        ax.plot(plot_data, label='Cls Loss', linewidth=2)
    if 'train/dfl_loss' in data and data['train/dfl_loss']:
        losses = data['train/dfl_loss']
        plot_data = smooth_curve(losses, window) if smooth else losses
        ax.plot(plot_data, label='DFL Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Precision/Recall曲线
    ax = axes[0, 1]
    if 'metrics/precision' in data and data['metrics/precision']:
        prec = data['metrics/precision']
        plot_data = smooth_curve(prec, window) if smooth else prec
        ax.plot(plot_data, label='Precision', linewidth=2, color='blue')
    if 'metrics/recall' in data and data['metrics/recall']:
        rec = data['metrics/recall']
        plot_data = smooth_curve(rec, window) if smooth else rec
        ax.plot(plot_data, label='Recall', linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. mAP@0.5曲线
    ax = axes[1, 0]
    if 'metrics/mAP50' in data and data['metrics/mAP50']:
        map50 = data['metrics/mAP50']
        plot_data = smooth_curve(map50, window) if smooth else map50
        ax.plot(plot_data, label='mAP@0.5', linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('mAP@0.5 During Training')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. mAP@0.5:0.95曲线
    ax = axes[1, 1]
    if 'metrics/mAP50-95' in data and data['metrics/mAP50-95']:
        map50_95 = data['metrics/mAP50-95']
        plot_data = smooth_curve(map50_95, window) if smooth else map50_95
        ax.plot(plot_data, label='mAP@0.5:0.95', linewidth=2, color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5:0.95')
    ax.set_title('mAP@0.5:0.95 During Training')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {output_path}")


def plot_loss_components(
    data: dict[str, list],
    title: str,
    output_path: Path
):
    """绘制详细的loss分解图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if 'train/box_loss' in data and data['train/box_loss']:
        ax.plot(data['train/box_loss'], label='Box Loss', linewidth=1.5, alpha=0.8)
    if 'train/cls_loss' in data and data['train/cls_loss']:
        ax.plot(data['train/cls_loss'], label='Classification Loss', linewidth=1.5, alpha=0.8)
    if 'train/dfl_loss' in data and data['train/dfl_loss']:
        ax.plot(data['train/dfl_loss'], label='Distribution Focal Loss', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{title} - Loss Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss components saved: {output_path}")


def compare_experiments(
    experiment_paths: dict[str, Path],
    metric: str,
    output_path: Path,
    title: str
):
    """对比多个实验的同一指标"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (exp_name, exp_path) in enumerate(experiment_paths.items()):
        results_csv = exp_path / "results.csv"
        if not results_csv.exists():
            continue

        data = parse_results_csv(results_csv)
        if metric in data and data[metric]:
            ax.plot(data[metric], label=exp_name, linewidth=2, color=colors[i % len(colors)])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{title} - {metric}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved: {output_path}")


def main():
    """主函数"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 查找所有实验的results.csv
    experiments_dir = PROJECT_ROOT / "experiments"

    if experiments_dir.exists():
        # 绘制每个实验的训练曲线
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            results_csv = exp_dir / "results.csv"
            if results_csv.exists():
                data = parse_results_csv(results_csv)
                plot_training_curves(
                    data,
                    title=f"Training Curves - {exp_dir.name}",
                    output_path=OUTPUT_DIR / f"{exp_dir.name}_curves.png"
                )

        # 绘制对比曲线
        experiment_paths = {
            exp.name: exp
            for exp in experiments_dir.iterdir()
            if exp.is_dir() and (exp / "results.csv").exists()
        }

        if len(experiment_paths) > 1:
            compare_experiments(
                experiment_paths,
                metric='metrics/mAP50',
                output_path=OUTPUT_DIR / "map50_comparison.png",
                title="mAP@0.5 Comparison"
            )

    print(f"\nTraining curves saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
