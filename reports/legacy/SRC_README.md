# src 目录说明

## 子目录职责

- `01_baseline/`：基础训练实验
- `02_training_strategy/`：训练策略实验
- `04_semi_supervised/`：半监督流程
- `utils/`：通用工具脚本（划分、合并、评估、增强）

## 统一约定

- 所有脚本优先提供 `main()` 入口
- 路径统一基于项目根目录构造
- 工具类脚本尽量无副作用导入（避免 import 即执行）

## 快速检查

- 冒烟检查：
  - `python src/smoke_test_ultralytics.py`
