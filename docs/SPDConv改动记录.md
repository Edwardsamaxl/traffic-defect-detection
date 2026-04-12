# SPD-Conv 改动记录

> 日期: 2026-04-09
> 项目: traffic-defect-detection
> 目标: 在 YOLOv8s + CBAM 基础上添加 SPD-Conv 下采样优化

---

## 一、改动概述

本次改动在 `ultralytics-main/` 中集成了 SPD-Conv 模块，用于替代传统下采样，减少细粒度信息丢失。

### 核心原理

**传统下采样 (stride=2)**:
```
输入: [B, C, H, W] → [B, C/4, H/2, W/2]  ← 信息丢失!
```

**SPD-Conv**:
```
输入: [B, C, H, W]
     ↓ Space-to-Depth (4象限拼接)
     [B, C*4, H/2, W/2]  ← 所有信息保留在通道中
     ↓ 非步长 3x3 conv
输出: [B, C/4, H/2, W/2]  ← 信息不丢失!
```

---

## 二、修改的文件清单

### 1. `ultralytics-main/ultralytics/nn/modules/conv.py`

**改动位置**: 文件末尾添加新类

**添加的内容**:
```python
class SPDConv(nn.Module):
    """SPD-Conv: Space-to-Depth + 非步长卷积"""
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, scale=2):
        super().__init__()
        if s != scale:
            raise ValueError(f"SPDConv: stride={s} must equal scale={scale}")
        self.scale = scale
        spd_c = c1 * (scale ** 2)
        self.conv = nn.Conv2d(spd_c, c2, k, stride=1, padding=(k // 2), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # Space-to-Depth 实现 (scale=2 优化版本)
        if self.scale == 2:
            x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        else:
            # 通用实现
            B, C, H, W = x.shape
            new_c = C * (self.scale ** 2)
            new_h = H // self.scale
            new_w = W // self.scale
            x = x.reshape(B, C, new_h, self.scale, new_w, self.scale)
            x = x.permute(0, 3, 5, 1, 2, 4).reshape(B, new_c, new_h, new_w)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)
```

---

### 2. `ultralytics-main/ultralytics/nn/modules/__init__.py`

**改动位置 1**: `from .conv import (...)` 部分

**添加**:
```python
    SPDConv,
```

**改动位置 2**: `__all__` 部分

**添加**:
```python
    "SPDConv",
```

---

### 3. `ultralytics-main/ultralytics/nn/tasks.py`

**改动位置 1**: imports 部分

**添加**:
```python
from ultralytics.nn.modules import (
    SPDConv,      # 新增
    CBAM,         # 新增 (原本存在于 conv.py 但 tasks.py 没有 import)
    SpatialAttention,   # 新增
    ChannelAttention,   # 新增
    ...
)
```

**改动位置 2**: `parse_model` 函数中，在 `elif m is Concat:` 之前添加

**添加**:
```python
        elif m is CBAM:
            c1, c2 = ch[f], ch[f]  # CBAM输出通道数=输入通道数
            args = [c1]
        elif m is SPDConv:
            # SPDConv 通道流: c1 -> c1*scale^2 -> c2
            # args 格式: [c2, k, s, scale]
            c1 = ch[f]  # 输入通道
            c2 = args[0]  # 目标输出通道
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]  # [c1, c2, k, s, scale]
```

---

### 4. `ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_spdconv.yaml`

**改动**: 无需修改，已存在的配置文件

**内容概要**:
- Backbone 第1、4层使用 SPDConv 替代标准 Conv
- 保持 CBAM 注意力机制
- Neck 和 Head 部分不变

---

## 三、备份文件

备份位置: `src/experiments/spd_conv/backups/`

**注意**: 此备份在重新克隆后可能已更新为干净版本

---

## 四、已知问题与解决方案

### 问题 1: `install_spd_conv.py` 脚本有 bug

**现象**: 脚本只添加了 `SPDConv` 到 `__all__`，但没有添加到 `from .conv import (...)` 部分

**解决方案**: 手动修复，手动在 `__init__.py` 和 `tasks.py` 中添加导入

### 问题 2: `parse_model` 返回的 Sequential 不支持多输入

**现象**: 直接调用 `parse_model` 后执行 forward 会报错 `TypeError: cat() received an invalid combination of arguments`

**解决方案**: 使用 `YOLO('yaml路径')` API 代替，它内部有正确的处理机制

**正确用法**:
```python
from ultralytics import YOLO
model = YOLO('ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_spdconv.yaml')
results = model(x)  # 正常工作
```

**错误用法**:
```python
from ultralytics.nn.tasks import parse_model
model, _ = parse_model(d, 3)
y = model(x)  # 会报错！
```

---

## 五、恢复方法

### 方法 1: 使用备份恢复

```bash
cd E:\PycharmProjects\traffic-defect-detection
python src/experiments/spd_conv/install_spd_conv.py restore
```

### 方法 2: 重新克隆 (推荐)

如果备份也被污染，执行:

```bash
# 1. 备份 YAML
cp ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_spdconv.yaml /tmp/

# 2. 删除并重新克隆
rm -rf ultralytics-main
git clone https://github.com/ultralytics/ultralytics ultralytics-main

# 3. 重新安装 SPD-Conv (需手动修复脚本)
python src/experiments/spd_conv/install_spd_conv.py install
# 然后手动修复 __init__.py 和 tasks.py (见第六节)

# 4. 恢复 YAML
cp /tmp/yolov8s_cbam_spdconv.yaml ultralytics-main/ultralytics/cfg/models/v8/
```

---

## 六、手动修复步骤 (如果 install_spd_conv.py 失败)

### 1. 修复 `ultralytics-main/ultralytics/nn/modules/__init__.py`

在 `from .conv import (...)` 中添加 `SPDConv`:
```python
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    # ... 其他导入 ...
    SpatialAttention,
    SPDConv,  # 添加这一行
)
```

在 `__all__` 中添加:
```python
    "SPDConv",
```

### 2. 修复 `ultralytics-main/ultralytics/nn/tasks.py`

在 imports 中添加:
```python
from ultralytics.nn.modules import (
    SPDConv,  # 添加
    CBAM,     # 添加
    SpatialAttention,  # 添加
    ChannelAttention,  # 添加
    # ... 其他导入 ...
)
```

在 `parse_model` 函数的 `elif m is Concat:` 之前添加:
```python
        elif m is CBAM:
            c1, c2 = ch[f], ch[f]
            args = [c1]
        elif m is SPDConv:
            c1 = ch[f]
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
```

---

## 七、测试命令

```bash
# 测试模型加载
cd E:\PycharmProjects\traffic-defect-detection
python src/experiments/exp07_spd_conv/test_model.py

# 训练
python src/experiments/exp07_spd_conv/train_spd_conv.py
```

---

## 八、相关文件路径

| 文件 | 路径 |
|------|------|
| SPD-Conv 模块 | `ultralytics-main/ultralytics/nn/modules/conv.py` |
| 模块导出 | `ultralytics-main/ultralytics/nn/modules/__init__.py` |
| 模型解析 | `ultralytics-main/ultralytics/nn/tasks.py` |
| 实验配置 | `ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_spdconv.yaml` |
| 安装脚本 | `src/experiments/spd_conv/install_spd_conv.py` |
| 备份目录 | `src/experiments/spd_conv/backups/` |
| 训练脚本 | `src/experiments/exp07_spd_conv/train_spd_conv.py` |
| 测试脚本 | `src/experiments/exp07_spd_conv/test_model.py` |
