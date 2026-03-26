"""
工具模块
"""
from .config import (
    PROJECT_ROOT,
    DATA_ROOT,
    EXPERIMENTS_ROOT,
    REPORTS_ROOT,
    OUTPUT_ROOT,
    CLASSES,
    NUM_CLASSES,
    get_device,
    get_device_name,
    get_memory_info,
    ensure_dirs,
)

from .trainer import (
    ExperimentRunner,
    quick_train,
    train_and_compare,
)

from .evaluator import (
    EnhancedEvaluator,
    batch_evaluate,
)

from .wandb_integration import (
    WandbLogger,
    ExperimentTracker,
    init_wandb,
)

from .analysis import (
    DatasetAnalyzer,
    compare_datasets,
)

__all__ = [
    # Config
    "PROJECT_ROOT",
    "DATA_ROOT",
    "EXPERIMENTS_ROOT",
    "REPORTS_ROOT",
    "OUTPUT_ROOT",
    "CLASSES",
    "NUM_CLASSES",
    "get_device",
    "get_device_name",
    "get_memory_info",
    "ensure_dirs",
    # Trainer
    "ExperimentRunner",
    "quick_train",
    "train_and_compare",
    # Evaluator
    "EnhancedEvaluator",
    "batch_evaluate",
    # Wandb
    "WandbLogger",
    "ExperimentTracker",
    "init_wandb",
    # Analysis
    "DatasetAnalyzer",
    "compare_datasets",
]
