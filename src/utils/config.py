"""
统一配置管理 - 所有脚本从这里读取项目路径和常量
"""
from pathlib import Path
from typing import Literal
import torch

# ========== 项目根目录 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
REPORTS_ROOT = PROJECT_ROOT / "reports"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# ========== 数据集路径 ==========
NEU_DET_ROOT = DATA_ROOT / "NEU-DET"
NEU_DET_SEMI_ROOT = DATA_ROOT / "NEU-DET-semi"

# 数据集配置
DATASETS = {
    "neu": PROJECT_ROOT / "datasets" / "neu.yaml",
    "neu_seed": PROJECT_ROOT / "datasets" / "neu_seed.yaml",
    "neu_copy_paste": PROJECT_ROOT / "datasets" / "neu_copy_paste.yaml",
    "neu_merge": PROJECT_ROOT / "datasets" / "neu_merge.yaml",
}

# ========== 模型路径 ==========
PRETRAINED_MODELS = {
    "yolov8n": PROJECT_ROOT / "yolov8n.pt",
    "yolov8s": PROJECT_ROOT / "yolov8s.pt",
    "yolov8m": PROJECT_ROOT / "yolov8m.pt",
    "yolo26n": PROJECT_ROOT / "src" / "yolo26n.pt",
}

# ========== 缺陷类别 ==========
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)

# ========== 设备 ==========
def get_device() -> str:
    """自动选择设备"""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def get_device_name() -> str:
    """获取设备名称"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def get_memory_info() -> dict:
    """获取显存信息"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(total - reserved, 2),
        }
    return {"total_gb": 0, "allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}

# ========== 训练常量 ==========
TRAIN_DEFAULTS = {
    "imgsz": 640,
    "epochs": 200,
    "patience": 50,
    "batch": 4,
    "workers": 8,
    "device": get_device(),
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}

# ========== 数据增强 ==========
AUG_DEFAULTS = {
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "blur": 0.0,
    "erode": 0.0,
}

# ========== 伪标签常量 ==========
PSEUDO_LABEL_DEFAULTS = {
    "base_conf": 0.65,
    "adaptive_lambda": 0.25,
    "iou_match": 0.6,
    "conservative_conf": 0.8,
    "standard_conf": 0.7,
}

# ========== 评估常量 ==========
EVAL_DEFAULTS = {
    "conf": 0.001,
    "iou": 0.6,
    "imgsz": 640,
    "split": "test",
}

# ========== 工具函数 ==========
def ensure_dirs(*dirs: Path) -> None:
    """确保目录存在"""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_latest_run(project: Path, name_prefix: str = "") -> Path | None:
    """获取最新的训练 run 目录"""
    runs_dir = project / "runs" / "detect"
    if not runs_dir.exists():
        return None
    runs = sorted(runs_dir.glob(f"{name_prefix}*"), key=lambda x: x.stat().st_mtime)
    return runs[-1] if runs else None

def get_best_weight(exp_dir: Path) -> Path | None:
    """从实验目录获取最佳权重"""
    if not exp_dir.exists():
        return None
    weights = exp_dir / "weights"
    if not weights.exists():
        return None
    best = weights / "best.pt"
    if best.exists():
        return best
    # 尝试其他命名
    for name in ["new-best.pt", "best-cosine.pt", "last.pt"]:
        p = weights / name
        if p.exists():
            return p
    return None
