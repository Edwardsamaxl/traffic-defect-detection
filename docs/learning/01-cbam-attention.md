# 01 - CBAM 注意力机制

> CBAM（Convolutional Block Attention Module）是一种轻量级注意力模块，通过通道注意力和空间注意力的顺序组合增强特征表示能力。

## 1. 学习目标

学完本章后，你将理解：
- CBAM 的两个核心组件：CAM 和 SAM
- 通道注意力和空间注意力的实现原理
- CBAM 在 YOLOv8 中的集成方式
- CBAM 与其他注意力机制的对比

## 2. 核心内容

### 2.1 CBAM 概述

CBAM 由 Woo 等人于 2018 年提出（ECCV），包含两个顺序连接的子模块：

| 模块 | 名称 | 作用 |
|------|------|------|
| CAM | Channel Attention Module | 重新校准通道权重，筛选重要特征通道 |
| SAM | Spatial Attention Module | 重新校准空间权重，聚焦关键区域 |

```
输入特征图 → CAM → SAM → 输出增强特征图
```

### 2.2 CAM（通道注意力）

**核心思想**：不是所有通道都同等重要，为重要通道分配更高权重。

**输入**：`[B, C, H, W]` 的特征图
**输出**：`[B, C, H, W]` 的加权特征图

**实现流程**：

```
x [B, C, H, W]
  │
  ├─► AvgPool(H,W → 1,1) ──► [B, C, 1, 1]
  │                              ↓
  └─► MaxPool(H,W → 1,1) ──► [B, C, 1, 1]
              │                    │
              └──────┬─────────────┘
                     ▼ concat
              [B, 2C, 1, 1]
                     │
                     ▼ 1x1Conv (2C → C)
              [B, C, 1, 1]
                     │
                     ▼ Sigmoid
              Mc [B, C, 1, 1]  ← 每通道一个权重
                     │
                     ▼ x × Mc (广播乘法)
输出 [B, C, H, W]
```

**AvgPool vs MaxPool 的区别**：

| 池化方式 | 提取的信息 | 代表意义 |
|----------|-----------|----------|
| AvgPool | 平均响应强度 | 整体一般化特征 |
| MaxPool | 最大响应强度 | 最显著特征 |

**关键代码**：

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)       # 全局平均池化
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)  # 1x1卷积
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))
```

### 2.3 SAM（空间注意力）

**核心思想**：不是所有空间位置都同等重要，聚焦在关键区域。

**输入**：`[B, C, H, W]` 的特征图
**输出**：`[B, C, H, W]` 的加权特征图

**实现流程**：

```
x [B, C, H, W]
  │
  ├─► AvgPool(C → 1) ──► [B, 1, H, W]  ← 沿通道压缩
  │
  └─► MaxPool(C → 1) ──► [B, 1, H, W]
              │                │
              └──────┬─────────┘
                     ▼ concat
              [B, 2, H, W]
                     │
                     ▼ 7x7Conv (2 → 1)
              [B, 1, H, W]
                     │
                     ▼ Sigmoid
              Ms [B, 1, H, W]  ← 每空间位置一个权重
                     │
                     ▼ x × Ms (广播乘法)
输出 [B, C, H, W]
```

**为什么用 7x7 卷积核？**

- 大卷积核提供更大的感受野
- 7x7 可覆盖 49 个像素的邻域信息
- 能判断一个位置"周围是什么样的"

**关键代码**：

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([
            torch.mean(x, 1, keepdim=True),
            torch.max(x, 1, keepdim=True)[0]], 1)))
```

### 2.4 CAM 与 SAM 的对比

| 维度 | CAM（通道注意力） | SAM（空间注意力） |
|------|------------------|-----------------|
| 池化维度 | 压缩 H×W → 1 | 压缩 C → 1 |
| 输出权重维度 | `[B, C, 1, 1]` | `[B, 1, H, W]` |
| 学到的内容 | 哪些通道重要 | 哪些位置重要 |
| 融合方式 | 1x1Conv (2C→C) | 7x7Conv (2→1) |

### 2.5 CBAM 在 YOLOv8 中的集成

CBAM 被插入到 backbone 的三个特征层之后：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]   # P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, CBAM, []]           # ← CBAM 在 P2/4 后
  - [-1, 1, Conv, [256, 3, 2]] # P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, CBAM, []]           # ← CBAM 在 P3/8 后
  - [-1, 1, Conv, [512, 3, 2]] # P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, CBAM, []]           # ← CBAM 在 P4/16 后
  - [-1, 1, Conv, [1024, 3, 2]]# P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
```

**注意**：CBAM 不改变通道数，只在通道和空间维度重新加权。

### 2.6 cfg 文件 vs 数据集配置文件

| 文件类型 | 作用 | 示例 |
|----------|------|------|
| 模型 cfg | 定义模型内部结构（层、通道、连接） | `yolov8s_cbam.yaml` |
| 数据集 yaml | 定义数据路径、类别名称 | `neu.yaml` |

```yaml
# neu.yaml - 数据集配置
path: E:/PycharmProjects/traffic-defect-detection/data/NEU-DET
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

## 3. 关键源文件

| 功能 | 文件 |
|------|------|
| CBAM 模块实现 | `ultralytics-main/ultralytics/nn/modules/conv.py` |
| CBAM 模型配置 | `ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam.yaml` |
| CBAM 训练脚本 | `src/02_training_strategy/train_cbam.py` |

## 4. 本章小结

- CBAM 由通道注意力（CAM）和空间注意力（SAM）顺序组成
- CAM 通过 AvgPool + MaxPool 压缩空间维度，学习通道权重
- SAM 通过沿通道压缩并用 7x7 卷积融合空间信息，学习位置权重
- CBAM 插入 backbone 的 P2、P3、P4 特征层之后
- cfg 文件定义模型结构，yaml 文件定义数据集

---

## 下一步

← 返回上一章

→ [02-cnn-basics.md](02-cnn-basics.md) CNN 基础知识
