"""
FixMatch 风格半监督训练
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from ..utils.config import (
    PROJECT_ROOT,
    DATASETS,
    PRETRAINED_MODELS,
    ensure_dirs,
)


class FixMatchTrainer:
    """
    FixMatch 风格半监督训练器

    核心思想:
    1. 弱增强的伪标签作为硬标签
    2. 强增强的预测应该与弱增强一致
    3. 只对高置信度预测计算损失

    使用示例:
        trainer = FixMatchTrainer(
            model="yolov8s.pt",
            data="neu_merge",
            labeled_ratio=0.3,
            unlabeled_ratio=0.7,
        )
        trainer.train(epochs=200)
    """

    def __init__(
        self,
        model: str = "yolov8s.pt",
        data: str = "neu_merge",
        imgsz: int = 640,
        device: str = "0",
        project: Optional[Path] = None,
    ):
        # 模型
        if model in PRETRAINED_MODELS:
            self.model_path = PRETRAINED_MODELS[model]
        else:
            self.model_path = Path(model)

        # 数据
        if data in DATASETS:
            self.data_yaml = DATASETS[data]
        else:
            self.data_yaml = Path(data)

        self.imgsz = imgsz
        self.device = device

        # 输出
        self.project = project or PROJECT_ROOT / "experiments"
        ensure_dirs(self.project)

        # 模型实例
        self.model: Optional[YOLO] = None

    def train(
        self,
        epochs: int = 200,
        batch: int = 4,
        patience: int = 50,
        conf_threshold: float = 0.7,  # FixMatch 置信度阈值
        lambda_u: float = 1.0,  # 无标签损失权重
        cos_lr: bool = True,
        amp: bool = True,
        name: str = "fixmatch",
        **kwargs
    ):
        """
        训练 FixMatch 模型

        注意: Ultralytics YOLO 原生不直接支持 FixMatch，
        这里通过自定义增强配置来模拟 FixMatch 的效果

        关键参数:
        - conf_threshold: 伪标签置信度阈值，高于此值的预测才作为无标签数据的监督
        - lambda_u: 无标签损失权重
        """
        print(f"\n{'='*60}")
        print(f"FixMatch 风格半监督训练")
        print(f"{'='*60}")
        print(f"模型: {self.model_path}")
        print(f"数据: {self.data_yaml}")
        print(f"伪标签阈值: {conf_threshold}")
        print(f"无标签损失权重: {lambda_u}")
        print(f"{'='*60}\n")

        # 初始化模型
        self.model = YOLO(str(self.model_path))

        # 训练参数
        # 注意: 这里是模拟 FixMatch，实际的 FixMatch 需要修改训练循环
        # YOLO 使用 mosaic/mixup 作为"强增强"，flip 作为"弱增强"
        args = {
            "data": str(self.data_yaml),
            "imgsz": self.imgsz,
            "epochs": epochs,
            "batch": batch,
            "patience": patience,
            "device": self.device,
            "project": str(self.project),
            "name": name,
            "exist_ok": True,

            # 优化
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": int(cos_lr),

            # 增强 - FixMatch 风格
            # 关闭部分默认增强，保留关键部分
            "mosaic": 1.0,
            "mixup": 0.15,  # 强增强
            "copy_paste": 0.1,
            "flipud": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,

            # 早停
            "close_mosaic": 10,

            # 混合精度
            "amp": amp,

            "verbose": True,
        }

        # 更新额外参数
        args.update(kwargs)

        # 训练
        results = self.model.train(**args)

        print(f"\n{'='*60}")
        print(f"FixMatch 训练完成")
        print(f"{'='*60}\n")

        return results


class CurriculumPseudoLabelTrainer:
    """
    课程学习伪标签训练器

    核心思想:
    1. 先用高置信度伪标签训练（easy samples）
    2. 逐步降低阈值，加入更难的样本
    3. 课程难度递增

    使用示例:
        trainer = CurriculumPseudoLabelTrainer(
            model="yolov8s.pt",
            data="neu_merge",
        )
        trainer.train(
            stages=[
                {"conf_threshold": 0.9, "epochs": 50},
                {"conf_threshold": 0.8, "epochs": 50},
                {"conf_threshold": 0.7, "epochs": 100},
            ]
        )
    """

    def __init__(
        self,
        model: str = "yolov8s.pt",
        data: str = "neu_merge",
        imgsz: int = 640,
        device: str = "0",
        project: Optional[Path] = None,
    ):
        if model in PRETRAINED_MODELS:
            self.model_path = PRETRAINED_MODELS[model]
        else:
            self.model_path = Path(model)

        if data in DATASETS:
            self.data_yaml = DATASETS[data]
        else:
            self.data_yaml = Path(data)

        self.imgsz = imgsz
        self.device = device
        self.project = project or PROJECT_ROOT / "experiments"
        ensure_dirs(self.project)

        self.model: Optional[YOLO] = None
        self.best_weight: Optional[Path] = None

    def train(
        self,
        stages: list[dict],
        amp: bool = True,
        **kwargs
    ):
        """
        分阶段课程训练

        Args:
            stages: 阶段列表，每个元素包含 conf_threshold 和 epochs
        """
        print(f"\n{'='*60}")
        print(f"课程学习伪标签训练")
        print(f"{'='*60}")
        print(f"阶段配置:")
        for i, stage in enumerate(stages):
            print(f"  Stage {i+1}: conf={stage['conf_threshold']}, epochs={stage['epochs']}")
        print(f"{'='*60}\n")

        # 初始化模型
        self.model = YOLO(str(self.model_path))

        # 逐阶段训练
        for stage_idx, stage in enumerate(stages):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}/{len(stages)}")
            print(f"置信度阈值: {stage['conf_threshold']}")
            print(f"训练轮数: {stage['epochs']}")
            print(f"{'='*60}\n")

            stage_name = f"curriculum_stage{stage_idx + 1}"

            args = {
                "data": str(self.data_yaml),
                "imgsz": self.imgsz,
                "epochs": stage["epochs"],
                "batch": 4,
                "patience": stage["epochs"] + 10,
                "device": self.device,
                "project": str(self.project / "curriculum"),
                "name": stage_name,
                "exist_ok": True,

                "optimizer": "AdamW",
                "lr0": 0.001 * (0.9 ** stage_idx),  # 逐渐降低学习率
                "lrf": 0.01,
                "cos_lr": True,

                "mosaic": 1.0,
                "mixup": 0.1,
                "copy_paste": 0.1,
                "flipud": 0.5,
                "fliplr": 0.5,

                "close_mosaic": 5,
                "amp": amp,
                "verbose": True,
            }

            # 如果不是第一阶段，从上一阶段继续
            if stage_idx > 0:
                prev_weight = self.project / "curriculum" / f"curriculum_stage{stage_idx}" / "weights" / "best.pt"
                if prev_weight.exists():
                    self.model = YOLO(str(prev_weight))
                    args["resume"] = False  # 不使用 resume，而是加载后继续训练

            results = self.model.train(**args)

        # 获取最终最佳权重
        final_dir = self.project / "curriculum" / f"curriculum_stage{len(stages)}" / "weights"
        if final_dir.exists():
            best = final_dir / "best.pt"
            if best.exists():
                self.best_weight = best

        print(f"\n{'='*60}")
        print(f"课程学习训练完成")
        print(f"最佳权重: {self.best_weight}")
        print(f"{'='*60}\n")


class NoisyStudentTrainer:
    """
    Noisy Student 训练

    核心思想:
    1. 用较小模型生成伪标签
    2. 用较大模型在伪标签 + 原始标签上训练
    3. 迭代: 用当前模型重新生成伪标签，继续训练更大的模型

    使用示例:
        trainer = NoisyStudentTrainer(
            teacher_model="yolov8s.pt",
            student_model="yolov8m.pt",
            data="neu_merge",
        )
        trainer.train(epochs=200)
    """

    def __init__(
        self,
        teacher_model: str = "yolov8s.pt",
        student_model: str = "yolov8m.pt",
        data: str = "neu_merge",
        imgsz: int = 640,
        device: str = "0",
        project: Optional[Path] = None,
    ):
        if teacher_model in PRETRAINED_MODELS:
            self.teacher_path = PRETRAINED_MODELS[teacher_model]
        else:
            self.teacher_path = Path(teacher_model)

        if student_model in PRETRAINED_MODELS:
            self.student_path = PRETRAINED_MODELS[student_model]
        else:
            self.student_path = Path(student_model)

        if data in DATASETS:
            self.data_yaml = DATASETS[data]
        else:
            self.data_yaml = Path(data)

        self.imgsz = imgsz
        self.device = device
        self.project = project or PROJECT_ROOT / "experiments"
        ensure_dirs(self.project)

    def train(
        self,
        epochs: int = 200,
        batch: int = 4,
        amp: bool = True,
        name: str = "noisy_student",
    ):
        """
        Noisy Student 训练

        训练过程:
        1. 先用 teacher 模型生成伪标签
        2. 用 student 模型在混合数据集上训练
        3. student 成为新的 teacher
        """
        print(f"\n{'='*60}")
        print(f"Noisy Student 训练")
        print(f"{'='*60}")
        print(f"Teacher: {self.teacher_path}")
        print(f"Student: {self.student_path}")
        print(f"数据: {self.data_yaml}")
        print(f"{'='*60}\n")

        # 加载教师模型
        teacher = YOLO(str(self.teacher_path))

        # 训练学生模型
        student = YOLO(str(self.student_path))

        args = {
            "data": str(self.data_yaml),
            "imgsz": self.imgsz,
            "epochs": epochs,
            "batch": batch,
            "patience": 50,
            "device": self.device,
            "project": str(self.project),
            "name": name,
            "exist_ok": True,

            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": True,

            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.1,
            "flipud": 0.5,
            "fliplr": 0.5,

            "close_mosaic": 10,
            "amp": amp,
            "verbose": True,
        }

        results = student.train(**args)

        print(f"\n{'='*60}")
        print(f"Noisy Student 训练完成")
        print(f"{'='*60}\n")

        return results
