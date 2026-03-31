"""
实验结果对比表生成脚本
"""
import csv
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "thesis" / "tables"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


@dataclass
class ExperimentResult:
    """单个实验的结果"""
    name: str
    model_path: str
    precision: float
    recall: float
    map50: float
    map75: float
    map50_95: float
    per_class_ap: dict[str, float]
    config: dict


class ResultsCollector:
    """收集和管理所有实验结果"""

    def __init__(self):
        self.results: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        """添加实验结果"""
        self.results.append(result)

    def get_supervised_baseline(self) -> Optional[ExperimentResult]:
        """获取监督学习基线"""
        for r in self.results:
            if "baseline" in r.name.lower() and "semi" not in r.name.lower():
                return r
        return None

    def get_semi_supervised(self) -> list[ExperimentResult]:
        """获取半监督实验结果"""
        return [r for r in self.results if "semi" in r.name.lower()]

    def get_ablation_studies(self) -> dict[str, ExperimentResult]:
        """获取消融实验结果"""
        ablation = {}
        for r in self.results:
            name_lower = r.name.lower()
            if "ablation" in name_lower or "adaptive" in name_lower or "consistency" in name_lower:
                ablation[r.name] = r
        return ablation

    def to_csv(self, output_path: Path):
        """导出为CSV格式"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 表头
            header = ['Experiment', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
            writer.writerow(header)

            # 数据行
            for r in self.results:
                writer.writerow([
                    r.name,
                    f"{r.precision:.4f}",
                    f"{r.recall:.4f}",
                    f"{r.map50:.4f}",
                    f"{r.map75:.4f}",
                    f"{r.map50_95:.4f}",
                ])

        print(f"Results exported to: {output_path}")

    def to_latex(self, output_path: Path):
        """导出为LaTeX表格格式"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Experimental Results}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Method & Precision & Recall & mAP@0.5 & mAP@0.5:0.95 \\\\\n")
            f.write("\\midrule\n")

            for r in self.results:
                f.write(f"{r.name} & {r.precision:.4f} & {r.recall:.4f} & {r.map50:.4f} & {r.map50_95:.4f} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"LaTeX table exported to: {output_path}")


def parse_ultralytics_results(results_json: Path) -> dict:
    """解析Ultralytics的results.json文件"""
    if not results_json.exists():
        return {}

    with open(results_json) as f:
        data = json.load(f)

    return data


def collect_experiment_results() -> ResultsCollector:
    """收集所有实验结果"""
    collector = ResultsCollector()

    # 遍历实验目录
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue

        # 查找权重文件
        weight_files = list(exp_dir.glob("weights/*.pt"))
        if not weight_files:
            continue

        # 读取训练参数
        args_file = exp_dir / "args.yaml"
        config = {}
        if args_file.exists():
            import yaml
            with open(args_file) as f:
                config = yaml.safe_load(f)

        # 收集结果（这里需要实际运行评估）
        # 注意：实际使用时应该调用model.val()获取结果
        result = ExperimentResult(
            name=exp_dir.name,
            model_path=str(weight_files[0]),
            precision=0.0,  # 待填充
            recall=0.0,
            map50=0.0,
            map75=0.0,
            map50_95=0.0,
            per_class_ap={cls: 0.0 for cls in CLASSES},
            config=config,
        )
        collector.add_result(result)

    return collector


def generate_comparison_table():
    """生成对比表格"""
    collector = collect_experiment_results()

    # 导出CSV
    collector.to_csv(OUTPUT_DIR / "experiment_results.csv")

    # 导出LaTeX
    collector.to_latex(OUTPUT_DIR / "experiment_results.tex")

    return collector


def create_ablation_table():
    """创建消融实验表格"""
    # 消融实验应该包含：
    # 1. 基线（无伪标签）
    # 2. 标准伪标签（固定阈值）
    # 3. 自适应阈值
    # 4. 翻转一致性
    # 5. 自适应 + 一致性

    ablation_data = [
        ("Baseline (Supervised)", 0.0, 0.0, 0.0, 0.0),
        ("+ Standard Pseudo-labels", 0.0, 0.0, 0.0, 0.0),
        ("+ Adaptive Threshold", 0.0, 0.0, 0.0, 0.0),
        ("+ Flip Consistency", 0.0, 0.0, 0.0, 0.0),
        ("Full (Adaptive + Consistency)", 0.0, 0.0, 0.0, 0.0),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "ablation_study.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'])
        for row in ablation_data:
            writer.writerow(row)

    # LaTeX格式
    with open(OUTPUT_DIR / "ablation_study.tex", 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study on Semi-supervised Components}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Configuration & Precision & Recall & mAP@0.5 & mAP@0.5:0.95 \\\\\n")
        f.write("\\midrule\n")
        for row in ablation_data:
            f.write(f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Ablation table saved to: {OUTPUT_DIR / 'ablation_study.csv'}")


def create_per_class_comparison():
    """创建每类AP对比表"""
    # 各方法在每类上的mAP@0.5对比
    data = {
        "Method": ["Baseline", "Semi-supervised", "Semi+Adaptive", "Semi+Full"],
        "crazing": [0.0, 0.0, 0.0, 0.0],
        "inclusion": [0.0, 0.0, 0.0, 0.0],
        "patches": [0.0, 0.0, 0.0, 0.0],
        "pitted_surface": [0.0, 0.0, 0.0, 0.0],
        "rolled-in_scale": [0.0, 0.0, 0.0, 0.0],
        "scratches": [0.0, 0.0, 0.0, 0.0],
        "mAP@0.5": [0.0, 0.0, 0.0, 0.0],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "per_class_comparison.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        for i in range(len(data["Method"])):
            row = {k: data[k][i] for k in data.keys()}
            writer.writerow(row)

    # LaTeX格式
    with open(OUTPUT_DIR / "per_class_comparison.tex", 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per-class mAP@0.5 Comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & crazing & inclusion & patches & pitted & rolled & scratches \\\\\n")
        f.write("\\midrule\n")
        for i, method in enumerate(data["Method"]):
            vals = [data[c][i] for c in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]]
            f.write(f"{method} & " + " & ".join([f"{v:.4f}" for v in vals]) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Per-class comparison saved to: {OUTPUT_DIR / 'per_class_comparison.csv'}")


if __name__ == "__main__":
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 生成对比表格
    generate_comparison_table()

    # 创建消融实验表格
    create_ablation_table()

    # 创建每类对比表
    create_per_class_comparison()

    print("\n表格生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
