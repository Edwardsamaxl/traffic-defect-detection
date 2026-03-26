# Traffic Defect Detection

钢材表面缺陷检测项目，基于 YOLOv8 的目标检测，支持多种训练策略和半监督学习方法。

## 项目结构

```
traffic-defect-detection/
├── train.py                 # 统一训练入口
├── evaluate.py              # 统一评估入口
├── generate_pseudo_labels.py # 伪标签生成入口
├── analyze_data.py          # 数据集分析入口
├── src/
│   ├── cfg/                # 训练策略配置
│   │   └── __init__.py     # 预定义策略 (baseline, cosine, semi_*, 等)
│   ├── semi/               # 半监督学习模块
│   │   ├── pseudo_label_generator.py  # 伪标签生成器
│   │   └── fixmatch.py     # FixMatch/课程学习/NoisyStudent
│   ├── utils/              # 工具模块
│   │   ├── config.py       # 统一配置管理
│   │   ├── trainer.py      # 训练执行器
│   │   ├── evaluator.py    # 增强评估器
│   │   ├── wandb_integration.py  # Wandb 集成
│   │   └── analysis.py     # 数据集分析
│   ├── webapp/             # Web 应用 (FastAPI)
│   └── runs/               # 训练输出
├── datasets/               # 数据集 YAML 配置
├── experiments/            # 训练权重和输出
├── reports/                # 评估报告和分析结果
└── data/                  # 数据集文件
```

## 快速开始

### 训练

```bash
# 使用预定义策略训练
python train.py --strategy baseline_s_advanced

# 指定额外参数
python train.py --strategy baseline --epochs 100 --batch 8

# 训练并对比多个策略
python train.py --strategies baseline cosine_100 heavy_aug

# 列出所有可用策略
python train.py --list
```

### 评估

```bash
# 评估单个模型
python evaluate.py --exp baseline --model experiments/baseline/weights/best.pt --analyze

# 批量评估
python evaluate.py --batch
```

### 伪标签生成

```bash
# 标准伪标签（固定阈值）
python generate_pseudo_labels.py --method standard --model experiments/baseline_seed/weights/best.pt

# 自适应阈值
python generate_pseudo_labels.py --method adaptive --model experiments/baseline_seed/weights/best.pt

# 翻转一致性
python generate_pseudo_labels.py --method consistency --model experiments/baseline_seed/weights/best.pt
```

### 数据分析

```bash
# 分析数据集
python analyze_data.py --data data/NEU-DET

# 对比多个数据集
python analyze_data.py --compare --datasets data/NEU-DET data/NEU-DET-semi --names original semi
```

## 可用训练策略

### 监督学习
| 策略名 | 说明 |
|--------|------|
| `baseline` | YOLOv8s 默认参数 |
| `baseline_s_advanced` | 增强版基线（cosine LR + 优化增强） |
| `ablation_no_aug` | 无增强消融实验 |
| `res_640` / `res_1024` / `res_1280` | 分辨率消融 |
| `cosine_30` / `cosine_100` | Cosine LR 对比 |
| `heavy_aug` | 强增强策略 |
| `light_aug` | 弱增强策略 |
| `copy_paste` | Copy-Paste 离线增强 |
| `yolov8m_baseline` | YOLOv8m 基线 |

### 半监督学习
| 策略名 | 说明 |
|--------|------|
| `seed_supervised` | 仅用 seed 数据监督训练 |
| `semi_adaptive` | 自适应阈值伪标签 |
| `semi_adaptive_conservative` | 保守阈值伪标签 |
| `semi_fixmatch` | FixMatch 风格训练 |
| `curriculum` | 课程学习伪标签 |

## 伪标签生成方法

| 方法 | 说明 |
|------|------|
| `standard` | 固定置信度阈值 |
| `adaptive` | 基于类别 AP 的自适应阈值 |
| `consistency` | 翻转一致性筛选 |
| `adaptive_consistency` | 自适应 + 一致性组合 |
| `uncertainty` | 基于不确定性的筛选 |

## 预定义配置

### 评估配置
- `conf=0.001`: 低置信度阈值用于完整性能评估
- `iou=0.6`: 标准 IoU 阈值
- `split=test`: 测试集评估

### 训练配置
- `epochs=200`: 完整训练
- `patience=50`: 早停耐心值
- `amp=True`: 混合精度训练
- `cos_lr=True`: 余弦学习率

## 缺陷类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | crazing | 龟裂/细裂纹 |
| 1 | inclusion | 夹杂物 |
| 2 | patches | 片状缺陷 |
| 3 | pitted_surface | 麻点表面 |
| 4 | rolled-in_scale | 轧入氧化皮 |
| 5 | scratches | 划痕 |

## 开发

### 添加新策略

在 `src/cfg/__init__.py` 中添加新的 `TrainConfig`:

```python
"my_strategy": TrainConfig(
    name="my_strategy",
    model="yolov8s.pt",
    data="neu",
    epochs=200,
    batch=4,
    cos_lr=True,
    # ... 其他参数
),
```

### 添加新伪标签方法

在 `src/semi/pseudo_label_generator.py` 中添加新方法:

```python
def generate_my_method(self, unlabeled_dir, output_dir, ...):
    # 实现伪标签生成逻辑
    pass
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- Ultralytics (本地源码: `ultralytics-main/`)
- Wandb (可选, 用于实验追踪)

## License

MIT
