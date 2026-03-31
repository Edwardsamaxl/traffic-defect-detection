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

### 2. 运行训练（四个版本一起跑）

```python
!python src/kaggle/kaggle_01_baseline_640.py
!python src/kaggle/kaggle_02_baseline_1024.py
!python src/kaggle/kaggle_03_copypaste_640.py
!python src/kaggle/kaggle_04_copypaste_1024.py
```

## 训练配置

| 脚本 | 模型 | Epochs | 分辨率 | Batch | 数据集 |
|------|------|--------|--------|-------|--------|
| `kaggle_01_baseline_640.py` | baseline_640 | 200 | 640 | 4 | neu.yaml |
| `kaggle_02_baseline_1024.py` | baseline_1024 | 200 | 1024 | 2 | neu.yaml |
| `kaggle_03_copypaste_640.py` | copypaste_640 | 200 | 640 | 4 | copy-paste |
| `kaggle_04_copypaste_1024.py` | copypaste_1024 | 200 | 1024 | 2 | copy-paste |

## 输出

训练结果保存在:
```
/kaggle/working/traffic-defect-detection/experiments/
├── baseline_640/weights/best.pt
├── baseline_1024/weights/best.pt
├── copypaste_640/weights/best.pt
└── copypaste_1024/weights/best.pt
```
