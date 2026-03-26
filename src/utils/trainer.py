"""
统一训练执行器 - 一行命令启动各种训练策略
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .config import (
    PROJECT_ROOT,
    EXPERIMENTS_ROOT,
    DATASETS,
    PRETRAINED_MODELS,
    get_device,
    ensure_dirs,
)


class ExperimentRunner:
    """
    统一训练执行器

    使用示例:
        runner = ExperimentRunner("baseline_s_advanced")
        runner.train()
        runner.eval()
    """

    def __init__(
        self,
        strategy_name: str,
        from_cfg,
        project: Optional[Path] = None,
        extra_args: Optional[dict] = None,
    ):
        """
        初始化执行器

        Args:
            strategy_name: 策略名称，用于命名实验
            from_cfg: TrainConfig 对象
            project: 实验输出目录，默认使用 EXPERIMENTS_ROOT
            extra_args: 额外参数，会覆盖 cfg 中的值
        """
        self.strategy_name = strategy_name
        self.cfg = from_cfg
        self.extra_args = extra_args or {}

        # 实验目录
        self.project = project or EXPERIMENTS_ROOT
        ensure_dirs(self.project)

        # 解析模型路径
        model_path = self.cfg.model
        if model_path in PRETRAINED_MODELS:
            self.model_path = PRETRAINED_MODELS[model_path]
        else:
            self.model_path = Path(model_path)

        # 解析数据路径
        data_key = self.cfg.data
        if data_key in DATASETS:
            self.data_yaml = DATASETS[data_key]
        else:
            self.data_yaml = Path(data_key)

        # 模型实例
        self.model: Optional[YOLO] = None
        self.best_weight: Optional[Path] = None

        # 训练结果
        self.results = None
        self.train_time = 0.0

    def _build_args(self) -> dict:
        """构建训练参数"""
        args = {
            # 模型和数据
            "model": str(self.model_path),
            "data": str(self.data_yaml),

            # 训练参数
            "imgsz": self.cfg.imgsz,
            "epochs": self.cfg.epochs,
            "patience": self.cfg.patience,
            "batch": self.cfg.batch,
            "workers": self.cfg.workers,
            "device": self.cfg.device,

            # 优化器
            "optimizer": self.cfg.optimizer,
            "lr0": self.cfg.lr0,
            "lrf": self.cfg.lrf,
            "momentum": self.cfg.momentum,
            "weight_decay": self.cfg.weight_decay,

            # 学习率
            "cos_lr": int(self.cfg.cos_lr),
            "warmup_epochs": self.cfg.warmup_epochs,
            "warmup_bias_lr": self.cfg.warmup_bias_lr,

            # 损失
            "box": self.cfg.box,
            "cls": self.cfg.cls,
            "dfl": self.cfg.dfl,

            # 增强
            "mosaic": self.cfg.mosaic,
            "mixup": self.cfg.mixup,
            "copy_paste": self.cfg.copy_paste,
            "flipud": self.cfg.flipud,
            "fliplr": self.cfg.fliplr,
            "hsv_h": self.cfg.hsv_h,
            "hsv_s": self.cfg.hsv_s,
            "hsv_v": self.cfg.hsv_v,
            "degrees": self.cfg.degrees,
            "translate": self.cfg.translate,
            "scale": self.cfg.scale,

            # 早停
            "close_mosaic": self.cfg.close_mosaic,

            # 混合精度
            "amp": self.cfg.amp,

            # 输出
            "project": str(self.project),
            "name": self.strategy_name,
            "exist_ok": True,
            "verbose": True,
            "save": True,
            "save_period": -1,
        }

        # 覆盖额外参数
        args.update(self.extra_args)

        return args

    def train(self, resume: bool = False) -> dict:
        """
        执行训练

        Args:
            resume: 是否从最后一个 checkpoint 恢复

        Returns:
            训练结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始训练: {self.strategy_name}")
        print(f"{'='*60}")
        print(f"模型: {self.model_path}")
        print(f"数据: {self.data_yaml}")
        print(f"实验目录: {self.project / self.strategy_name}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # 初始化模型
        self.model = YOLO(str(self.model_path))

        # 构建参数
        args = self._build_args()

        if resume:
            args["resume"] = True

        # 训练
        self.results = self.model.train(**args)

        self.train_time = time.time() - start_time

        # 获取最佳权重
        exp_dir = self.project / self.strategy_name
        weights_dir = exp_dir / "weights"
        if weights_dir.exists():
            best = weights_dir / "best.pt"
            if best.exists():
                self.best_weight = best

        print(f"\n{'='*60}")
        print(f"训练完成: {self.strategy_name}")
        print(f"耗时: {self.train_time:.1f}s ({self.train_time/60:.1f}min)")
        print(f"最佳权重: {self.best_weight}")
        print(f"{'='*60}\n")

        return self._format_results()

    def eval(
        self,
        weight_path: Optional[Path] = None,
        conf: float = 0.001,
        iou: float = 0.6,
        tta: bool = False,
    ) -> dict:
        """
        执行评估

        Args:
            weight_path: 权重路径，默认使用训练的最佳权重
            conf: 置信度阈值
            iou: IoU 阈值
            tta: 是否使用 TTA

        Returns:
            评估结果字典
        """
        if weight_path:
            self.best_weight = weight_path

        if not self.best_weight or not self.best_weight.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.best_weight}")

        print(f"\n{'='*60}")
        print(f"开始评估: {self.strategy_name}")
        print(f"权重: {self.best_weight}")
        print(f"{'='*60}\n")

        # 加载模型
        model = YOLO(str(self.best_weight))

        # 评估
        metrics = model.val(
            data=str(self.data_yaml),
            imgsz=self.cfg.imgsz,
            conf=conf,
            iou=iou,
            augment=tta,
            verbose=True,
        )

        results = metrics.results_dict

        print(f"\n{'='*60}")
        print(f"评估结果:")
        print(f"  Precision:  {results.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:     {results.get('metrics/recall(B)', 0):.4f}")
        print(f"  mAP@0.5:    {results.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:95: {results.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"{'='*60}\n")

        return results

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.6,
        save: bool = True,
        **kwargs
    ):
        """使用最佳权重进行预测"""
        if not self.best_weight:
            raise FileNotFoundError("未找到权重文件")

        model = YOLO(str(self.best_weight))
        return model.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            save=save,
            **kwargs
        )

    def _format_results(self) -> dict:
        """格式化训练结果"""
        if self.results is None:
            return {}

        r = self.results
        return {
            "strategy": self.strategy_name,
            "train_time_s": self.train_time,
            "best_weight": str(self.best_weight) if self.best_weight else None,
            "final_map50": getattr(r, "map50", None),
            "final_map50_95": getattr(r, "map", None),
        }

    def print_summary(self):
        """打印实验摘要"""
        print(f"\n{'='*60}")
        print(f"实验摘要: {self.strategy_name}")
        print(f"{'='*60}")
        print(f"策略配置:")
        for k, v in self.cfg.to_dict().items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")


def quick_train(strategy_name: str, **kwargs):
    """
    快速训练 - 一行命令启动训练

    使用示例:
        quick_train("baseline_s_advanced")
        quick_train("ablation_no_aug", epochs=100)
        quick_train("semi_adaptive", batch=8)
    """
    from src.cfg import get_strategy

    cfg = get_strategy(strategy_name)
    if cfg is None:
        raise ValueError(f"未知的策略: {strategy_name}")

    runner = ExperimentRunner(strategy_name, cfg, extra_args=kwargs)
    return runner.train()


def train_and_compare(*strategy_names: str):
    """
    训练并对比多个策略

    使用示例:
        train_and_compare("baseline", "cosine_100", "heavy_aug")
    """
    from src.cfg import get_strategy
    import pandas as pd

    results = []
    for name in strategy_names:
        cfg = get_strategy(name)
        if cfg is None:
            print(f"[WARN] 跳过未知策略: {name}")
            continue

        runner = ExperimentRunner(name, cfg)
        result = runner.train()
        runner.eval()

        results.append({
            "strategy": name,
            "train_time_min": result["train_time_s"] / 60,
            "mAP50": result.get("final_map50", 0),
            "mAP50_95": result.get("final_map50_95", 0),
        })

    # 打印对比表格
    print("\n" + "="*80)
    print("策略对比")
    print("="*80)
    print(f"{'策略':<30} {'时间(min)':<12} {'mAP@0.5':<12} {'mAP@0.5:95':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['strategy']:<30} {r['train_time_min']:<12.1f} {r['mAP50']:<12.4f} {r['mAP50_95']:<12.4f}")
    print("="*80 + "\n")

    return results
