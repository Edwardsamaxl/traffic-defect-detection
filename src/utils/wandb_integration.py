"""
Wandb 实验追踪集成
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .config import PROJECT_ROOT


class WandbLogger:
    """
    Wandb 日志记录器

    使用示例:
        logger = WandbLogger("baseline", {"epochs": 100, "batch": 4})
        logger.log_metrics({"train/loss": 0.5, "val/map50": 0.7})
        logger.finish()
    """

    def __init__(
        self,
        project_name: str,
        config: Optional[dict] = None,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        """
        初始化 Wandb logger

        Args:
            project_name: wandb 项目名称
            config: 配置参数字典
            run_name: 运行名称，默认使用时间戳
            entity: wandb entity (username/team)
        """
        if not HAS_WANDB:
            raise ImportError("wandb 未安装，请运行: pip install wandb")

        self.project_name = project_name
        self.entity = entity

        # 初始化 wandb
        wandb.init(
            project=project_name,
            name=run_name,
            entity=entity,
            config=config,
            dir=str(PROJECT_ROOT / "wandb"),
        )

        self.run = wandb.run
        self.enabled = True

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None):
        """记录指标"""
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_image(self, name: str, image: Any, step: Optional[int] = None):
        """记录图像"""
        if not self.enabled:
            return
        wandb.log({name: wandb.Image(image)}, step=step)

    def log_table(self, name: str, table: Any):
        """记录表格"""
        if not self.enabled:
            return
        wandb.log({name: table})

    def finish(self):
        """结束 wandb 运行"""
        if self.enabled:
            wandb.finish()
            self.enabled = False


class ExperimentTracker:
    """
    实验追踪器 - 同时支持 wandb 和本地 CSV

    使用示例:
        tracker = ExperimentTracker("baseline", {"strategy": "cosine"})
        tracker.log("train/loss", 0.5)
        tracker.log("val/map50", 0.7, step=100)
        tracker.save_summary({"best_map50": 0.75})
    """

    def __init__(
        self,
        experiment_name: str,
        config: Optional[dict] = None,
        use_wandb: bool = True,
        log_dir: Optional[Path] = None,
    ):
        self.experiment_name = experiment_name
        self.config = config or {}
        self.use_wandb = use_wandb and HAS_WANDB

        # 本地日志
        self.log_dir = log_dir or PROJECT_ROOT / "reports" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"

        # Wandb
        self.wandb_logger = None
        if self.use_wandb:
            try:
                self.wandb_logger = WandbLogger(
                    project_name="traffic-defect-detection",
                    config=self.config,
                    run_name=experiment_name,
                )
            except Exception as e:
                print(f"[WARN] Wandb 初始化失败: {e}")
                self.use_wandb = False

        # 指标记录
        self.metrics_history: list[dict] = []

    def log(self, key: str, value: Any, step: Optional[int] = None):
        """记录单个指标"""
        entry = {"key": key, "value": value, "step": step}
        self.metrics_history.append(entry)

        # Wandb
        if self.wandb_logger:
            self.wandb_logger.log_metrics({key: value}, step=step)

        # 本地
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None):
        """批量记录指标"""
        for key, value in metrics.items():
            self.log(key, value, step=step)

    def save_summary(self, summary: dict[str, Any]):
        """保存实验摘要"""
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "experiment": self.experiment_name,
                "config": self.config,
                "metrics": self.metrics_history,
                "summary": summary,
            }, f, indent=2)

        if self.wandb_logger:
            # Wandb summary
            for key, value in summary.items():
                wandb.define_metric(key)
                wandb.run.summary[key] = value

    def finish(self):
        """结束追踪"""
        if self.wandb_logger:
            self.wandb_logger.finish()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()


def init_wandb(project_name: str = "traffic-defect-detection"):
    """
    初始化 wandb（登录等）

    使用示例:
        init_wandb()
        wandb.finish()  # 确保没有残留
    """
    if not HAS_WANDB:
        print("[WARN] wandb 未安装，跳过初始化")
        return False

    # 检查是否已登录
    try:
        wandb.ensure_configured()
    except Exception:
        print("[WARN] wandb 未登录，请先运行: wandb login")
        return False

    return True
