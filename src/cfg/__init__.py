"""
训练策略配置 - 统一管理所有训练配置
"""
from dataclasses import dataclass, field
from typing import Literal

# ========== 训练阶段配置 ==========

@dataclass
class TrainConfig:
    """基础训练配置"""
    name: str
    model: str = "yolov8s.pt"
    data: str = "neu"
    imgsz: int = 640
    epochs: int = 200
    batch: int = 4
    patience: int = 50
    device: str = "0"
    workers: int = 8

    # 优化器
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # 学习率调度
    cos_lr: bool = False
    warmup_epochs: int = 3
    warmup_bias_lr: float = 0.1

    # 损失权重
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # 增强
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    flipud: float = 0.5
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    fl_gamma: float = 0.0  # Focal gamma
    blur: float = 0.0
    erode: float = 0.0

    # 半监督特有
    weak_aug: bool = False
    strong_aug: bool = False
    curriculum: bool = False

    # 早停
    close_mosaic: int = 10
    amp: bool = True

    # 额外参数
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and k != 'extra':
                result[k] = v
        result.update(self.extra)
        return result


# ========== 预定义训练策略 ==========

STRATEGIES = {

    # ---- 监督基线 ----
    "baseline": TrainConfig(
        name="baseline",
        model="yolov8s.pt",
        data="neu",
        epochs=120,
        batch=4,
        patience=50,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
    ),

    "baseline_s_advanced": TrainConfig(
        name="baseline_s_advanced",
        model="yolov8s.pt",
        data="neu",
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        amp=True,
        # 增强优化
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
    ),

    # ---- 消融实验 ----
    "ablation_no_aug": TrainConfig(
        name="ablation_no_aug",
        model="yolov8s.pt",
        data="neu",
        epochs=200,
        batch=4,
        patience=50,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        flipud=0.0,
        fliplr=0.0,
        close_mosaic=10,
        amp=True,
    ),

    "ablation_no_aug_scratch": TrainConfig(
        name="ablation_no_aug_scratch",
        model="yolov8s.pt",
        data="neu",
        epochs=200,
        batch=4,
        patience=50,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        flipud=0.0,
        fliplr=0.0,
        close_mosaic=10,
        cos_lr=True,
        amp=True,
    ),

    # ---- 分辨率消融 ----
    "res_640": TrainConfig(
        name="res_640",
        model="yolov8s.pt",
        data="neu",
        imgsz=640,
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    "res_1024": TrainConfig(
        name="res_1024",
        model="yolov8s.pt",
        data="neu",
        imgsz=1024,
        epochs=200,
        batch=2,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    # ---- 高分辨率 + 小 batch ----
    "res_1280": TrainConfig(
        name="res_1280",
        model="yolov8s.pt",
        data="neu",
        imgsz=1280,
        epochs=200,
        batch=1,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    # ---- 余弦退火对比 ----
    "cosine_30": TrainConfig(
        name="cosine_30",
        model="yolov8s.pt",
        data="neu",
        epochs=30,
        patience=20,
        batch=4,
        cos_lr=True,
        close_mosaic=5,
        amp=True,
    ),

    "cosine_100": TrainConfig(
        name="cosine_100",
        model="yolov8s.pt",
        data="neu",
        epochs=100,
        patience=30,
        batch=4,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    # ---- 大模型 ----
    "yolov8m_baseline": TrainConfig(
        name="yolov8m_baseline",
        model="yolov8m.pt",
        data="neu",
        epochs=150,
        batch=2,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    "yolov8m_res640": TrainConfig(
        name="yolov8m_res640",
        model="yolov8m.pt",
        data="neu",
        imgsz=640,
        epochs=150,
        batch=2,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    # ---- Copy-Paste 增强 ----
    "copy_paste": TrainConfig(
        name="copy_paste",
        model="yolov8s.pt",
        data="neu_copy_paste",
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    # ---- Seed 监督 ----
    "seed_supervised": TrainConfig(
        name="seed_supervised",
        model="yolov8s.pt",
        data="neu_seed",
        epochs=120,
        batch=4,
        patience=50,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
    ),

    "seed_supervised_advanced": TrainConfig(
        name="seed_supervised_advanced",
        model="yolov8s.pt",
        data="neu_seed",
        epochs=150,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    # ---- 半监督训练 ----
    "semi_adaptive": TrainConfig(
        name="semi_adaptive",
        model="yolov8s.pt",
        data="neu_merge",
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        # 半监督特有
        fl_gamma=1.5,
    ),

    "semi_adaptive_conservative": TrainConfig(
        name="semi_adaptive_conservative",
        model="yolov8s.pt",
        data="neu_merge",
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        fl_gamma=1.5,
    ),

    "semi_fixmatch": TrainConfig(
        name="semi_fixmatch",
        model="yolov8s.pt",
        data="neu_merge",
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        # FixMatch 特有
        weak_aug=True,
        strong_aug=True,
    ),

    # ---- 课程学习 ----
    "curriculum": TrainConfig(
        name="curriculum",
        model="yolov8s.pt",
        data="neu",
        epochs=200,
        batch=4,
        patience=50,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        curriculum=True,
    ),

    # ---- 更多增强变体 ----
    "heavy_aug": TrainConfig(
        name="heavy_aug",
        model="yolov8s.pt",
        data="neu",
        epochs=200,
        batch=4,
        patience=50,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.15,
        scale=0.6,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

    "light_aug": TrainConfig(
        name="light_aug",
        model="yolov8s.pt",
        data="neu",
        epochs=200,
        batch=4,
        patience=50,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        flipud=0.0,
        fliplr=0.3,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    ),

}


def get_strategy(name: str) -> TrainConfig | None:
    """获取指定策略配置"""
    return STRATEGIES.get(name)


def list_strategies() -> list[str]:
    """列出所有可用策略"""
    return list(STRATEGIES.keys())
