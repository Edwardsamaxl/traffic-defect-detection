# Kaggle 训练脚本

## 使用方法

### 1. 配置环境 Cell
```python
!git clone -b main https://github.com/Edwardsamaxl/traffic-defect-detection.git
%cd traffic-defect-detection
!pip uninstall -y ultralytics
!pip install ultralytics==8.4.9
!pip install opencv-python tqdm pyyaml
!wget -q -O yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 2. 运行训练（每次跑一个）

**基线模型**:
```python
!python src/kaggle/kaggle_01_baseline.py
```

**Stage4 (标准增强)**:
```python
!python src/kaggle/kaggle_02_stage4.py
```

**Stage9 (无增强消融)**:
```python
!python src/kaggle/kaggle_03_stage9.py
```

## 训练配置

| 脚本 | 模型 | Epochs | 增强 | 说明 |
|------|------|--------|------|------|
| `kaggle_01_baseline.py` | baseline_s | 200 | 默认 | 基线模型 |
| `kaggle_02_stage4.py` | stage4_overall | 200 | mosaic+flip | 标准增强 |
| `kaggle_03_stage9.py` | stage9_no_aug | 200 | 无 | 无增强对比 |

## 输出

训练结果保存在:
```
/kaggle/working/traffic-defect-detection/experiments/
├── baseline_s/
│   └── weights/best.pt
├── stage4_overall/
│   └── weights/best.pt
└── stage9_no_aug/
    └── weights/best.pt
```
