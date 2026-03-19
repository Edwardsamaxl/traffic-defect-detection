# 脚本重命名映射

- `src/test.py` -> `src/smoke_test_ultralytics.py`
- `src/utils/copy_paste.py` -> `src/utils/offline_copy_paste_augmentation.py`
- `src/utils/split_data.py` -> `src/utils/split_detection_dataset.py`
- `src/utils/merge_data.py` -> `src/utils/merge_semi_supervised_dataset.py`
- `src/utils/evaluation.py` -> `src/utils/evaluate_model.py`

- `src/04_semi_supervised/label_predict.py` -> `src/04_semi_supervised/generate_pseudo_labels.py`
- `src/04_semi_supervised/label_predict_conservative.py` -> `src/04_semi_supervised/generate_pseudo_labels_conservative.py`
- `src/04_semi_supervised/adaptive_label_predict.py` -> `src/04_semi_supervised/generate_pseudo_labels_adaptive.py`
- `src/04_semi_supervised/new_split_data.py` -> `src/04_semi_supervised/split_seed_unlabeled.py`
- `src/04_semi_supervised/new_split_data_conservative.py` -> `src/04_semi_supervised/split_seed_unlabeled_conservative.py`
- `src/04_semi_supervised/new_merge_data.py` -> `src/04_semi_supervised/merge_seed_pseudo_train_only.py`
- `src/04_semi_supervised/merge_data_adaptive.py` -> `src/04_semi_supervised/merge_dataset_adaptive.py`
- `src/04_semi_supervised/merge_data_conservative.py` -> `src/04_semi_supervised/merge_dataset_conservative.py`
- `src/04_semi_supervised/train_seed.py` -> `src/04_semi_supervised/train_seed_supervised.py`
