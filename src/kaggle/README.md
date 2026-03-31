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

**640基线**:
```python
!python src/kaggle/kaggle_01_baseline_640.py
```

**1024高分辨率**:
```python
!python src/kaggle/kaggle_02_baseline_1024.py
```

## 训练配置

| 脚本 | 模型 | Epochs | 分辨率 | Batch | 说明 |
|------|------|--------|--------|-------|------|
| `kaggle_01_baseline_640.py` | baseline_640 | 200 | 640 | 4 | 基线分辨率 |
| `kaggle_02_baseline_1024.py` | baseline_1024 | 200 | 1024 | 2 | 高分辨率 |

## 输出

训练结果保存在:
```
/kaggle/working/traffic-defect-detection/experiments/
├── baseline_640/
│   └── weights/best.pt
└── baseline_1024/
    └── weights/best.pt
```
