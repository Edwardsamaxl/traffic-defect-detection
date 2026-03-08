# traffic-defect-detection

本仓库用于交通/表面缺陷检测实验，当前采用本地 `ultralytics-main` 源码开发模式。

## 命名规范

- Python 文件统一使用 `snake_case.py`
- 文件名优先使用动词+对象，例如：
  - `generate_pseudo_labels.py`
  - `split_seed_unlabeled.py`
  - `merge_semi_supervised_dataset.py`
- 训练脚本统一用 `train_` 前缀，评估脚本统一用 `evaluate_` 前缀

## 本地开发环境

- 解释器：`.venv/Scripts/python.exe`
- 本地源码：`ultralytics-main`
- 编辑器解析路径已通过 `.vscode/settings.json` 和 `pyrightconfig.json` 配置

## 目录说明

- `src/`：训练、数据处理和评估脚本
- `datasets/`：数据集 YAML 配置
- `experiments/`：训练输出和权重
- `ultralytics-main/`：本地可修改的 Ultralytics 源码
