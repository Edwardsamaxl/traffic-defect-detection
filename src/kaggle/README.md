# Kaggle 训练指南

## 方式一: Notebooks (网页操作)

### 步骤1: 准备数据集
1. 登录 Kaggle
2. 上传 NEU-DET 数据集或使用已有数据集
3. 记下数据集路径 (通常在 `/kaggle/input/`)

### 步骤2: 创建Notebook
1. 点击 "New Notebook"
2. 选择 "Code" -> "Python"

### 步骤3: 上传脚本
1. 打开本目录下的 `kaggle_semi_supervised_train.py`
2. 复制全部内容到 Kaggle Notebook
3. 根据实际情况修改 `DATA_ROOT` 路径

### 步骤4: 运行
1. 点击 "Run All" 或逐个Cell运行
2. 等待训练完成
3. 下载结果 (weights/ 文件夹)

---

## 方式二: Kaggle API (本地操作)

### 安装Kaggle API
```bash
pip install kaggle
```

### 配置API密钥
1. 在 Kaggle -> Account -> API -> Create New Token
2. 下载 `kaggle.json`
3. 放到 `~/.kaggle/kaggle.json` (Linux/Mac) 或 `C:\Users\YourName\.kaggle\kaggle.json` (Windows)

### 上传和运行
```bash
# 创建新数据集
kaggle datasets create -p /path/to/your/dataset

# 上传Notebook
kaggle kernels push -p /path/to/notebook

# 或者在Kaggle网页上手动上传脚本
```

---

## 方式三: 本地调试 -> Kaggle运行

1. **本地调试脚本**:
```bash
cd E:\PycharmProjects\traffic-defect-detection
python src/kaggle/kaggle_semi_supervised_train.py
```

2. **验证能跑通后**, 复制脚本到 Kaggle Notebooks
3. **调整路径**后运行

---

## 数据集准备

### Kaggle数据集结构
```
/kaggle/input/
└── neu-det/
    ├── images/
    │   ├── train/
    │   │   ├── *.jpg
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── labels/
        ├── train/
        │   ├── *.txt
        │   └── ...
        └── test/
            └── ...
```

### 如果数据集不同
修改脚本中的 `DATA_ROOT`:
```python
DATA_ROOT = Path("/kaggle/input/your-dataset-name")
```

---

## 脚本说明

| 脚本 | 说明 |
|------|------|
| `kaggle_semi_supervised_train.py` | 主训练脚本 (包含完整流程) |
| `kaggle_ablation.py` | 消融实验脚本 |
| `kaggle_evaluation.py` | 评估脚本 |

---

## 运行时间预估

| 阶段 | 时间 |
|------|------|
| 数据准备 | ~5分钟 |
| 基线训练 (100 epochs) | ~30分钟 (P100) |
| 伪标签生成 | ~10分钟 |
| 半监督训练 (150 epochs) | ~45分钟 |
| **总计** | ~90分钟 |

---

## 输出

训练完成后，结果保存在:
```
/kaggle/working/outputs/experiments/
├── baseline/
│   └── weights/
│       └── best.pt
└── semi_supervised/
    └── weights/
        └── best.pt
```

下载这些权重到本地 `experiments/` 目录即可。

---

## 注意事项

1. **Kaggle免费GPU限制**: 最多连续运行9小时
2. **保存检查点**: 定期保存模型，避免中断
3. **内存**: 如果内存不足，减少 `batch_size`
4. **数据路径**: 确保 Kaggle 数据集路径正确
