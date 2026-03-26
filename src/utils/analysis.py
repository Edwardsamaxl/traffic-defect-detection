"""
数据分析工具 - 数据集统计、可视化、分布分析
"""
from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from .config import (
    PROJECT_ROOT,
    CLASSES,
    REPORTS_ROOT,
    ensure_dirs,
)


class DatasetAnalyzer:
    """
    数据集分析器

    使用示例:
        analyzer = DatasetAnalyzer("data/NEU-DET")
        analyzer.analyze_all()
        analyzer.plot_class_distribution()
        analyzer.plot_bbox_size_distribution()
        analyzer.generate_report()
    """

    def __init__(self, dataset_root: Path | str, name: Optional[str] = None):
        self.dataset_root = Path(dataset_root)
        self.name = name or self.dataset_root.name

        # 输出目录
        self.output_dir = REPORTS_ROOT / "dataset_analysis" / self.name
        ensure_dirs(self.output_dir)

        # 统计数据
        self.num_images = 0
        self.num_labels = 0
        self.class_counts = defaultdict(int)
        self.bbox_sizes = []
        self.bbox_aspect_ratios = []
        self.images_per_class = defaultdict(int)

    def analyze_split(self, split: str = "train") -> dict:
        """
        分析单个 split

        Args:
            split: 'train', 'val', 'test'

        Returns:
            统计结果字典
        """
        img_dir = self.dataset_root / "images" / split
        lbl_dir = self.dataset_root / "labels" / split

        if not img_dir.exists():
            return {}

        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        label_files = list(lbl_dir.glob("*.txt"))

        stats = {
            "split": split,
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "class_counts": defaultdict(int),
            "total_objects": 0,
        }

        for lbl_file in label_files:
            with open(lbl_file) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    stats["class_counts"][CLASSES[cls_id]] += 1
                    stats["total_objects"] += 1

                    # 收集 bbox 大小
                    self.bbox_sizes.append((w, h))
                    self.bbox_aspect_ratios.append(w / (h + 1e-9))

        return stats

    def analyze_all(self) -> dict:
        """
        分析所有 split

        Returns:
            完整统计结果
        """
        print(f"\n{'='*60}")
        print(f"数据集分析: {self.name}")
        print(f"{'='*60}\n")

        results = {}
        for split in ["train", "val", "test"]:
            stats = self.analyze_split(split)
            if stats:
                results[split] = stats
                self.num_images += stats["num_images"]
                self.num_labels += stats["num_labels"]

                for cls_name, count in stats["class_counts"].items():
                    self.class_counts[cls_name] += count

        self._print_summary(results)
        return results

    def _print_summary(self, results: dict):
        """打印统计摘要"""
        print(f"\n{'='*60}")
        print(f"统计摘要")
        print(f"{'='*60}")

        for split, stats in results.items():
            print(f"\n{split.upper()}:")
            print(f"  图像数: {stats['num_images']}")
            print(f"  标签数: {stats['num_labels']}")
            print(f"  总目标数: {stats['total_objects']}")

        print(f"\n类别分布:")
        for cls_name, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(count / max(self.class_counts.values()) * 30)
            print(f"  {cls_name:<20} {count:<6} {bar}")

        print(f"\n总目标数: {sum(self.class_counts.values())}")
        print(f"{'='*60}\n")

    def plot_class_distribution(self, save: bool = True):
        """绘制类别分布图"""
        if not self.class_counts:
            self.analyze_all()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 柱状图
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())

        ax1 = axes[0]
        bars = ax1.bar(range(len(classes)), counts, color="steelblue")
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha="right")
        ax1.set_ylabel("Instance Count")
        ax1.set_title("Class Distribution (Instance Count)")

        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=9)

        # 饼图
        ax2 = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=classes,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title("Class Distribution (Proportion)")

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "class_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存: {output_path}")

        plt.close()

    def plot_bbox_size_distribution(self, save: bool = True):
        """绘制 bbox 大小分布图"""
        if not self.bbox_sizes:
            self.analyze_all()

        widths = [s[0] for s in self.bbox_sizes]
        heights = [s[1] for s in self.bbox_sizes]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 宽度分布
        axes[0].hist(widths, bins=30, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("Normalized Width")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Bounding Box Width Distribution")
        axes[0].axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.3f}")
        axes[0].legend()

        # 高度分布
        axes[1].hist(heights, bins=30, color="coral", edgecolor="white")
        axes[1].set_xlabel("Normalized Height")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Bounding Box Height Distribution")
        axes[1].axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean: {np.mean(heights):.3f}")
        axes[1].legend()

        # 宽高比分布
        axes[2].hist(self.bbox_aspect_ratios, bins=30, color="seagreen", edgecolor="white")
        axes[2].set_xlabel("Aspect Ratio (W/H)")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Bounding Box Aspect Ratio Distribution")
        axes[2].axvline(np.mean(self.bbox_aspect_ratios), color="red", linestyle="--",
                       label=f"Mean: {np.mean(self.bbox_aspect_ratios):.3f}")
        axes[2].legend()

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "bbox_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存: {output_path}")

        plt.close()

    def plot_sample_images(self, num_samples: int = 16, split: str = "train"):
        """绘制样本图像"""
        img_dir = self.dataset_root / "images" / split
        lbl_dir = self.dataset_root / "labels" / split

        if not img_dir.exists():
            return

        image_files = list(img_dir.glob("*.jpg"))[:num_samples]

        cols = 4
        rows = (len(image_files) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes]

        for idx, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 读取标签
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                with open(lbl_path) as f:
                    h, w = img.shape[:2]
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id = int(parts[0])
                        x, y, bw, bh = map(float, parts[1:])

                        # 转换为像素坐标
                        x1 = int((x - bw/2) * w)
                        y1 = int((y - bh/2) * h)
                        x2 = int((x + bw/2) * w)
                        y2 = int((y + bh/2) * h)

                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, CLASSES[cls_id], (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(img_path.name[:30])

        # 隐藏多余的子图
        for idx in range(len(image_files), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        output_path = self.output_dir / f"sample_images_{split}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"样本图像已保存: {output_path}")
        plt.close()

    def generate_report(self, save: bool = True) -> dict:
        """生成分析报告"""
        if not self.class_counts:
            self.analyze_all()

        report = {
            "dataset": self.name,
            "num_images": self.num_images,
            "num_labels": self.num_labels,
            "class_counts": dict(self.class_counts),
            "total_objects": sum(self.class_counts.values()),
            "bbox_stats": {
                "mean_width": float(np.mean([s[0] for s in self.bbox_sizes])) if self.bbox_sizes else 0,
                "mean_height": float(np.mean([s[1] for s in self.bbox_sizes])) if self.bbox_sizes else 0,
                "mean_aspect_ratio": float(np.mean(self.bbox_aspect_ratios)) if self.bbox_aspect_ratios else 0,
            },
        }

        if save:
            report_path = self.output_dir / "analysis_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"报告已保存: {report_path}")

        # 同时绘制图表
        self.plot_class_distribution(save=save)
        self.plot_bbox_size_distribution(save=save)
        self.plot_sample_images(split="train", num_samples=16)

        return report


def compare_datasets(
    dataset_paths: list[Path | str],
    names: Optional[list[str]] = None,
):
    """
    对比多个数据集的统计信息

    Args:
        dataset_paths: 数据集路径列表
        names: 数据集名称列表
    """
    if names is None:
        names = [str(p) for p in dataset_paths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 收集数据
    all_class_counts = {}
    for ds_path, ds_name in zip(dataset_paths, names):
        analyzer = DatasetAnalyzer(ds_path, ds_name)
        analyzer.analyze_all()
        all_class_counts[ds_name] = analyzer.class_counts

    # 获取所有类别
    all_classes = set()
    for counts in all_class_counts.values():
        all_classes.update(counts.keys())

    # 绘制对比柱状图
    x = np.arange(len(all_classes))
    width = 0.8 / len(dataset_paths)

    ax1 = axes[0]
    for i, (ds_name, counts) in enumerate(all_class_counts.items()):
        values = [counts.get(cls, 0) for cls in sorted(all_classes)]
        ax1.bar(x + i * width, values, width, label=ds_name)

    ax1.set_xticks(x + width * (len(dataset_paths) - 1) / 2)
    ax1.set_xticklabels(sorted(all_classes), rotation=45, ha="right")
    ax1.set_ylabel("Instance Count")
    ax1.set_title("Class Distribution Comparison")
    ax1.legend()

    # 绘制总目标数对比
    ax2 = axes[1]
    total_counts = [sum(c.values()) for c in all_class_counts.values()]
    ax2.bar(names, total_counts, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax2.set_ylabel("Total Objects")
    ax2.set_title("Total Object Count Comparison")

    plt.tight_layout()

    output_path = REPORTS_ROOT / "dataset_analysis" / "comparison.png"
    ensure_dirs(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"对比图已保存: {output_path}")
    plt.close()
