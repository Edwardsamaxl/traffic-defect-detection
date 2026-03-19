# 半监督流程说明

本目录覆盖从数据拆分、伪标签生成到合并再训练的完整流程。

## 推荐流程

1. `split_seed_unlabeled.py` 或 `split_seed_unlabeled_conservative.py`
2. `train_seed_supervised.py`
3. `generate_pseudo_labels.py` / `generate_pseudo_labels_conservative.py` / `generate_pseudo_labels_adaptive.py`
4. `merge_seed_pseudo_train_only.py` 或 `merge_dataset_*.py`
5. `train_stage6_semi.py`

## 脚本命名解释

- `generate_*`：生成伪标签
- `split_*`：数据拆分
- `merge_*`：数据合并
- `train_*`：训练

## 注意事项

- 同名目录（如 `seed`、`unlabeled`）会被脚本写入
- 运行前确认数据目录和阈值配置符合当前实验目标
