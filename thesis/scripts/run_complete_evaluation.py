"""
完整评估脚本 - 用于生成论文所需的所有实验数据
支持评估多个实验并生成对比表格
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import numpy as np

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
THESIS_TABLES_DIR = PROJECT_ROOT / "thesis" / "tables"
THESIS_FIGURES_DIR = PROJECT_ROOT / "thesis" / "figures"

# 缺陷类别
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)

# 实验配置
EXPERIMENT_CONFIGS = {
    "baseline_s": {
        "model": "yolov8s.pt",
        "data": "neu.yaml",
        "description": "YOLOv8s监督学习基线",
    },
    "baseline_seed": {
        "model": "experiments/baseline_seed/weights/best.pt",
        "data": "neu_seed.yaml",
        "description": "Seed数据监督学习",
    },
    "stage4_overall": {
        "model": "experiments/stage4_overall/weights/best.pt",
        "data": "neu.yaml",
        "description": "优化后的监督学习",
    },
    "stage6_semi": {
        "model": "experiments/stage6_semi/weights/best.pt",
        "data": "neu_merge.yaml",
        "description": "半监督学习（自适应阈值）",
    },
}


@dataclass
class ModelResult:
    """单个模型的评估结果"""
    experiment_name: str
    model_path: Path
    data_yaml: Path

    # 总体指标
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map75: float = 0.0
    map50_95: float = 0.0

    # 每类AP
    per_class_ap: dict[str, float] = field(default_factory=dict)

    # 配置信息
    config: dict = field(default_factory=dict)

    # 状态
    evaluated: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment_name,
            "precision": self.precision,
            "recall": self.recall,
            "mAP@0.5": self.map50,
            "mAP@0.75": self.map75,
            "mAP@0.5:0.95": self.map50_95,
            "per_class_ap": self.per_class_ap,
            "evaluated": self.evaluated,
            "error": self.error,
        }


class ThesisEvaluator:
    """
    论文评估器 - 评估所有实验并生成论文所需的数据
    """

    def __init__(self):
        self.results: dict[str, ModelResult] = {}
        self.ultralytics_model = None

    def _load_ultralytics(self):
        """延迟加载ultralytics"""
        if self.ultralytics_model is None:
            from ultralytics import YOLO
            self.ultralytics_model = YOLO
        return self.ultralytics_model

    def evaluate_experiment(
        self,
        experiment_name: str,
        model_path: Path,
        data_yaml: Path,
        conf: float = 0.001,
        iou: float = 0.6,
        imgsz: int = 640,
        tta: bool = False,
    ) -> ModelResult:
        """
        评估单个实验

        Args:
            experiment_name: 实验名称
            model_path: 模型权重路径
            data_yaml: 数据集配置文件
            conf: 置信度阈值
            iou: NMS IoU阈值
            imgsz: 输入图像大小
            tta: 是否使用测试时增强

        Returns:
            ModelResult: 评估结果
        """
        result = ModelResult(
            experiment_name=experiment_name,
            model_path=model_path,
            data_yaml=data_yaml,
        )

        if not model_path.exists():
            result.error = f"Model not found: {model_path}"
            self.results[experiment_name] = result
            return result

        try:
            # 加载模型
            YOLO = self._load_ultralytics()
            model = YOLO(str(model_path))

            # 执行评估
            print(f"\n{'='*60}")
            print(f"评估: {experiment_name}")
            print(f"模型: {model_path}")
            print(f"数据: {data_yaml}")
            print(f"{'='*60}")

            metrics = model.val(
                data=str(data_yaml),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                augment=tta,
                verbose=True,
                split="test",
            )

            # 提取结果
            results_dict = metrics.results_dict

            result.precision = float(results_dict.get("metrics/precision(B)", 0))
            result.recall = float(results_dict.get("metrics/recall(B)", 0))
            result.map50 = float(results_dict.get("metrics/mAP50(B)", 0))
            result.map75 = float(results_dict.get("metrics/mAP75(B)", 0))
            result.map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0))

            # 提取每类AP
            for key, value in results_dict.items():
                if "metrics/mAP50(" in key and key.endswith(")"):
                    cls_name = key.split("(")[1].rstrip(")")
                    result.per_class_ap[cls_name] = float(value)

            result.evaluated = True

            print(f"\n结果:")
            print(f"  Precision: {result.precision:.4f}")
            print(f"  Recall:    {result.recall:.4f}")
            print(f"  mAP@0.5:   {result.map50:.4f}")
            print(f"  mAP@0.75:  {result.map75:.4f}")
            print(f"  mAP@0.5:0.95: {result.map50_95:.4f}")

        except Exception as e:
            result.error = str(e)
            print(f"[ERROR] 评估失败: {e}")

        self.results[experiment_name] = result
        return result

    def evaluate_all(self, use_existing: bool = True):
        """
        评估所有配置的实验

        Args:
            use_existing: 是否使用已缓存的结果
        """
        THESIS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

        for exp_name, config in EXPERIMENT_CONFIGS.items():
            model_path = PROJECT_ROOT / config["model"]
            data_yaml = PROJECT_ROOT / "datasets" / config["data"]

            # 检查是否已有结果
            result_cache = THESIS_TABLES_DIR / f"{exp_name}_result.json"
            if use_existing and result_cache.exists():
                print(f"\n[使用缓存] {exp_name}")
                with open(result_cache) as f:
                    cached = json.load(f)
                    result = ModelResult(**cached)
                    self.results[exp_name] = result
                    continue

            # 执行评估
            result = self.evaluate_experiment(
                experiment_name=exp_name,
                model_path=model_path,
                data_yaml=data_yaml,
            )

            # 缓存结果
            with open(result_cache, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

    def generate_results_table(self):
        """生成实验结果对比表"""
        output_csv = THESIS_TABLES_DIR / "experiment_results.csv"
        output_tex = THESIS_TABLES_DIR / "experiment_results.tex"

        rows = []
        for name, result in self.results.items():
            if result.error:
                continue
            rows.append({
                "Experiment": result.experiment_name,
                "Description": EXPERIMENT_CONFIGS.get(name, {}).get("description", ""),
                "Precision": f"{result.precision:.4f}",
                "Recall": f"{result.recall:.4f}",
                "mAP@0.5": f"{result.map50:.4f}",
                "mAP@0.75": f"{result.map75:.4f}",
                "mAP@0.5:0.95": f"{result.map50_95:.4f}",
            })

        # CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        # LaTeX
        with open(output_tex, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Experimental Results on NEU-DET Dataset}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write("Method & Precision & Recall & mAP@0.5 & mAP@0.75 & mAP@0.5:0.95 \\\\\n")
            f.write("\\midrule\n")

            for row in rows:
                f.write(f"{row['Description']} & {row['Precision']} & {row['Recall']} & "
                       f"{row['mAP@0.5']} & {row['mAP@0.75']} & {row['mAP@0.5:0.95']} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"\n结果表已生成:")
        print(f"  CSV: {output_csv}")
        print(f"  LaTeX: {output_tex}")

        return rows

    def generate_ablation_table(self):
        """生成消融实验表"""
        # 这里需要实际的消融实验结果
        # 模板数据结构

        ablation_data = [
            # (配置名称, Precision, Recall, mAP@0.5, mAP@0.5:0.95)
            ("Baseline (Supervised)", "0.0000", "0.0000", "0.0000", "0.0000"),
            ("+ Standard Pseudo-labels", "0.0000", "0.0000", "0.0000", "0.0000"),
            ("+ Adaptive Threshold", "0.0000", "0.0000", "0.0000", "0.0000"),
            ("+ Flip Consistency", "0.0000", "0.0000", "0.0000", "0.0000"),
            ("Full (Adaptive + Consistency)", "0.0000", "0.0000", "0.0000", "0.0000"),
        ]

        output_csv = THESIS_TABLES_DIR / "ablation_study.csv"
        output_tex = THESIS_TABLES_DIR / "ablation_study.tex"

        # CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Configuration', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'])
            writer.writerows(ablation_data)

        # LaTeX
        with open(output_tex, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Ablation Study on Semi-supervised Components}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Configuration & Precision & Recall & mAP@0.5 & mAP@0.5:0.95 \\\\\n")
            f.write("\\midrule\n")
            for row in ablation_data:
                f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"\n消融实验表已生成:")
        print(f"  CSV: {output_csv}")
        print(f"  LaTeX: {output_tex}")

    def generate_per_class_table(self):
        """生成每类AP对比表"""
        output_csv = THESIS_TABLES_DIR / "per_class_comparison.csv"
        output_tex = THESIS_TABLES_DIR / "per_class_comparison.tex"

        # 构建数据
        methods = list(self.results.keys())
        if not methods:
            print("[WARNING] 没有可用的评估结果")
            return

        header = ["Method"] + CLASSES + ["mAP@0.5"]
        rows = []

        for method in methods:
            result = self.results[method]
            if not result.evaluated:
                continue

            row = [EXPERIMENT_CONFIGS.get(method, {}).get("description", method)]
            for cls in CLASSES:
                ap = result.per_class_ap.get(cls, 0.0)
                row.append(f"{ap:.4f}")
            row.append(f"{result.map50:.4f}")
            rows.append(row)

        # CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        # LaTeX
        with open(output_tex, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Per-class mAP@0.5 Comparison}\n")
            f.write("\\begin{tabular}{l" + "c" * len(CLASSES) + "c}\n")
            f.write("\\toprule\n")

            # 表头
            f.write("Method")
            for cls in CLASSES[:4]:  # 简化显示
                f.write(f" & \\rotatebox{{90}}{{{cls}}}")
            f.write(" & \\rotatebox{90}{mAP@0.5} \\\\\n")
            f.write("\\midrule\n")

            for row in rows:
                method_name = row[0]
                # 截断过长的方法名
                if len(method_name) > 25:
                    method_name = method_name[:22] + "..."
                f.write(method_name)
                for val in row[1:]:  # 跳过方法名
                    f.write(f" & {val}")
                f.write(" \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"\n每类AP对比表已生成:")
        print(f"  CSV: {output_csv}")
        print(f"  LaTeX: {output_tex}")

    def generate_summary(self):
        """生成评估摘要"""
        output_json = THESIS_TABLES_DIR / "evaluation_summary.json"
        output_md = THESIS_TABLES_DIR / "evaluation_summary.md"

        summary = {
            "generated_at": datetime.now().isoformat(),
            "experiments": {},
        }

        for name, result in self.results.items():
            summary["experiments"][name] = result.to_dict()

        # JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        # Markdown
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write("# 实验评估摘要\n\n")
            f.write(f"生成时间: {summary['generated_at']}\n\n")

            f.write("## 总体结果\n\n")
            f.write("| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |\n")
            f.write("|--------|-----------|--------|---------|---------------|\n")

            for name, result in self.results.items():
                if result.error:
                    f.write(f"| {name} | ERROR: {result.error} | - | - | - |\n")
                else:
                    f.write(f"| {name} | {result.precision:.4f} | {result.recall:.4f} | "
                           f"{result.map50:.4f} | {result.map50_95:.4f} |\n")

            f.write("\n## 每类AP\n\n")
            for name, result in self.results.items():
                if not result.evaluated:
                    continue
                f.write(f"### {name}\n\n")
                for cls in CLASSES:
                    ap = result.per_class_ap.get(cls, 0.0)
                    f.write(f"- {cls}: {ap:.4f}\n")
                f.write("\n")

        print(f"\n摘要已生成:")
        print(f"  JSON: {output_json}")
        print(f"  Markdown: {output_md}")


def main():
    """主函数"""
    print("="*60)
    print("论文实验评估工具")
    print("="*60)

    evaluator = ThesisEvaluator()

    # 评估所有实验（如果已有缓存则跳过）
    evaluator.evaluate_all(use_existing=True)

    # 生成结果表
    evaluator.generate_results_table()

    # 生成消融实验表（模板）
    evaluator.generate_ablation_table()

    # 生成每类AP对比表
    evaluator.generate_per_class_table()

    # 生成摘要
    evaluator.generate_summary()

    print("\n" + "="*60)
    print("评估完成！")
    print(f"输出目录: {THESIS_TABLES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
