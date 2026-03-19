# utils 工具脚本说明

## 脚本列表

- `split_detection_dataset.py`：将全量数据划分为 train/val/test
- `merge_semi_supervised_dataset.py`：合并 seed + pseudo 并重划分
- `offline_copy_paste_augmentation.py`：离线 copy-paste 增强
- `evaluate_model.py`：对指定权重执行评估
- `common.py`：项目根目录等公共路径

## 使用建议

- 先执行划分类脚本，再执行训练脚本
- 合并类脚本会修改目标目录，建议先备份数据
- 评估前确认 `model_path` 和 `data_yaml` 指向正确
