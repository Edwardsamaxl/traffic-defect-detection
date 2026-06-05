# YOLOv8 自定义手册（CBAM / P2 / 模块扩展）

## 一、关键文件位置速查

| 用途 | 文件路径 |
|------|----------|
| CBAM / C2f 等模块实现 | `ultralytics-main/ultralytics/nn/modules/conv.py` |
| 模块注册表（`__init__.py`） | `ultralytics-main/ultralytics/nn/modules/__init__.py` |
| 模型解析器（YAML → PyTorch） | `ultralytics-main/ultralytics/nn/tasks.py` 的 `parse_model()` |
| YAML 模型配置目录 | `ultralytics-main/ultralytics/cfg/models/v8/` |
| 训练脚本示例 | `src/02_training_strategy/train_cbam.py` |

---

## 二、CBAM 注意力机制

### 2.1 代码实现

**`ultralytics-main/ultralytics/nn/modules/conv.py`**

| 类名 | 行号 | 作用 |
|------|------|------|
| `ChannelAttention` | ~514 | 通道注意力：全局平均池化 + 1x1 卷积 + Sigmoid |
| `SpatialAttention` | ~551 | 空间注意力：通道维度统计 + 卷积 + Sigmoid |
| `CBAM` | ~585 | 简化版：通道注意力 → 空间注意力，顺序执行 |
| `CBAMFull` | ~618 | 论文完整版：AvgPool+MaxPool + 共享 MLP（降维比 r=16） |

### 2.2 模块注册流程

要让 YAML 能识别 `CBAM`，需要经过三步：

1. **`conv.py` 中定义类**（已完成）
2. **`modules/__init__.py` 导入并导出**
   ```python
   from .conv import (
       CBAM,
       CBAMFull,
       # ...
   )
   __all__ = (
       # ...
       "CBAM",
       "CBAMFull",
       # ...
   )
   ```
3. **`tasks.py` 的 `parse_model()` 中处理参数**
   ```python
   elif m is CBAM:
       c1, c2 = ch[f], ch[f]  # 输出通道 = 输入通道
       args = [c1]
   elif m is CBAMFull:
       c1, c2 = ch[f], ch[f]
       args = [c1]
   ```

### 2.3 YAML 配置列表

全部位于 `ultralytics-main/ultralytics/cfg/models/v8/`：

| 文件 | 说明 |
|------|------|
| `yolov8s_cbam.yaml` | 标准版：P3、P4、P5 各加一层 CBAM |
| `yolov8s_cbam_p2p3p4p5.yaml` | P2、P3、P4、P5 都加 CBAM |
| `yolov8s_cbam_p2p3.yaml` | 只在 P2、P3 加 CBAM |
| `yolov8s_cbam_p2only.yaml` | 只在 P2 加 CBAM |
| `yolov8s_cbam_p3only.yaml` | 只在 P3 加 CBAM |
| `yolov8s_cbam_p3p4.yaml` | 只在 P3、P4 加 CBAM |
| `yolov8s_cbam_p4only.yaml` | 只在 P4 加 CBAM |
| `yolov8s_cbam_p5only.yaml` | 只在 P5 加 CBAM |
| `yolov8s_cbamfull_p2p3p4p5.yaml` | CBAMFull，P2-P5 |
| `yolov8s_cbamfull_p2p3p4.yaml` | CBAMFull，P2-P4 |
| `yolov8s_cbam_gfpn.yaml` | CBAM + BiFPN Neck |
| `yolov8s_cbam_spdconv.yaml` | CBAM + SPD-Conv |

### 2.4 在 YAML 中插入 CBAM

格式：`[-1, repeats, ModuleName, [args]]`

示例（在 P2 后插入）：
```yaml
backbone:
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]   # 2
  - [-1, 1, CBAM, []]            # 3 - 插入 CBAM
  - [-1, 1, Conv, [256, 3, 2]]  # 4-P3/8
```

**关键规则**：
- `CBAM` 的 `args` 为空列表 `[]`，因为 `parse_model` 会自动传入输入通道数 `c1`
- 插入 CBAM 后，**后续所有层索引 +1**，head 里的 `Concat` 引用也要同步调整

---

## 三、P2 检测层

### 3.1 原生 P2 检测模型

文件：`ultralytics-main/ultralytics/cfg/models/v8/yolov8-p2.yaml`

对比标准 YOLOv8，P2 模型的核心改动：

1. **head 多了一次上采样**：从 P3 再向上到 P2
2. **多了一条 Concat**：融合 backbone 的 P2 特征（层索引 2）
3. **Detect 层多一个输入**：`[[18, 21, 24, 27], 1, Detect, [nc]]` —— 4 个检测分支

```yaml
head:
  # FPN top-down: P5 → P4 → P3 → P2
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]   # cat backbone P4
  - [-1, 3, C2f, [512]]          # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]   # cat backbone P3
  - [-1, 3, C2f, [256]]          # 15 (P3/8-small)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]   # cat backbone P2（关键改动）
  - [-1, 3, C2f, [128]]          # 18 (P2/4-xsmall)

  # PAN bottom-up: P2 → P3 → P4 → P5
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]  # cat head P3
  - [-1, 3, C2f, [256]]          # 21 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]          # 24 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]   # cat head P5
  - [-1, 3, C2f, [1024]]         # 27 (P5/32-large)

  - [[18, 21, 24, 27], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
```

### 3.2 重要区分

现有的 `yolov8s_cbam_p2p3p4p5.yaml` 等文件**只加了 CBAM，但没有 P2 检测头**，Detect 仍然是：
```yaml
- [[19, 22, 25], 1, Detect, [nc]]  # Detect(P3, P4, P5) —— 只有3个分支
```

如需**CBAM + P2 检测头**，需新建 YAML，把 `yolov8-p2.yaml` 的 head 和 CBAM backbone 结合起来。

---

## 四、自定义 YOLO 的操作方法

### 4.1 场景 A：只改结构（已有模块的组合）

以"CBAM + P2 检测头"为例：

1. 新建 `yolov8s_cbam_p2.yaml`
2. 复制 `yolov8-p2.yaml` 的完整结构
3. 在 backbone 的 P2、P3、P4、P5 后插入 `CBAM, []`
4. **调整所有索引**：每插入一个 CBAM，后续层索引 +1
5. 修改 head 中的 `Concat` 引用，指向新的 backbone 索引
6. 训练时：`model = YOLO("yolov8s_cbam_p2.yaml")`

### 4.2 场景 B：添加全新的自定义模块

以自定义注意力 `MyAttention` 为例：

1. **实现模块**：在 `conv.py` 末尾添加
   ```python
   class MyAttention(nn.Module):
       def __init__(self, c1):
           super().__init__()
           # ...
       def forward(self, x):
           return x  # 输出通道数必须与输入相同
   ```
2. **注册到 `__init__.py`**：导入并加入 `__all__`
3. **注册到解析器**：`tasks.py` 中 import 并在 `parse_model()` 添加分支
   ```python
   elif m is MyAttention:
       c1, c2 = ch[f], ch[f]
       args = [c1]
   ```
4. **YAML 中使用**：`[-1, 1, MyAttention, []]`

### 4.3 场景 C：修改训练超参数

参考 `src/02_training_strategy/train_cbam.py`：

```python
from ultralytics import YOLO

model = YOLO("ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam.yaml")
model.train(
    data="datasets/neu.yaml",
    epochs=200,
    imgsz=640,
    batch=4,
    # ...
)
```

---

## 五、为什么 YAML 只有 backbone 和 head，neck 去哪了

**Neck 被合并进了 `head` 部分。**

在 YOLOv5 的 YAML 中有明确的 `backbone` / `neck` / `head` 三段式结构，但 YOLOv8 把 neck 和 head 合并了。

看 `yolov8.yaml` 的 `head`：
```yaml
head:
  # === Neck（FPN top-down）===
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]        # P3

  # === Neck（PAN bottom-up）===
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]        # P4

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]       # P5

  # === Head（Detect）===
  - [[15, 18, 21], 1, Detect, [nc]]
```

代码层面，`tasks.py` 的 `parse_model()` 直接拼接：
```python
for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
```

它不区分 neck 和 head，只认 `backbone` + `head` 两个列表。所以 neck 的物理存在就是 `head` 列表里 `Detect` 之前的那些层。

---

## 六、C2f 模块详解

### 6.1 C2f 和普通 Conv 的区别

| | 普通 `Conv` | `C2f` |
|--|------------|-------|
| 结构 | 单一卷积层（Conv + BN + Act） | CSP Bottleneck，含多个子分支 |
| 参数量 | 少 | 多（有多个 Bottleneck） |
| 梯度流 | 单一路径 | 丰富（chunk + 多分支 cascade） |
| 作用 | 特征提取 / 下采样 | 高效融合多尺度局部特征 |
| 典型位置 | Backbone 开头、下采样层 | Backbone 主体、Neck 融合后 |

### 6.2 C2f 内部结构

源码：`ultralytics-main/ultralytics/nn/modules/block.py` 第 289 行

```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)           # 隐藏通道数（默认 c2 的一半）
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)   # 1x1 卷积，输出分成两份
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 最后 1x1 融合所有分支
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))   # 沿通道分成两半：[y0, y1]
        y.extend(m(y[-1]) for m in self.m)   # y1 过 Bottleneck，输出追加到列表
        return self.cv2(torch.cat(y, 1))     # 所有分支 concat 后 1x1 融合
```

**数据流示意图（n=1 时）**：

```
输入 x (C1×H×W)
    │
    ▼
cv1: 1x1 Conv → 输出 2*C 通道
    │
    ├── chunk(2,1) ──┬──► y0 (C×H×W) ───────────────┐
    │                │                                │
    │                └──► y1 (C×H×W) ──► Bottleneck ─┤
    │                                                    │
    ◄────────────── concat(y0, y1, Bottleneck_out) ────┘
    │
    ▼
cv2: 1x1 Conv → 输出 C2 通道
```

**关键点**：
- `chunk(2, 1)` 把特征沿通道维度切成两半
- 其中一半（`y1`）过 `n` 个串联的 `Bottleneck`，另一半（`y0`）作为 shortcut 保留
- 最后把所有分支在通道维度 `concat`，再用 `1x1` 卷积融合
- `shortcut=True` 时，Bottleneck 内部也有残差连接

### 6.3 为什么用 C2f 而不是堆 Conv

YOLOv8 用 C2f 替换掉了 YOLOv5 的 C3，核心优势：

1. **梯度流更丰富**：C2f 把所有中间分支都保留并 concat，反向传播时梯度路径更多
2. **计算效率更高**：相同参数量下，C2f 比 C3 更快（论文称 "Faster Implementation"）
3. **特征融合更充分**：chunk + extend 的结构让每个 Bottleneck 都能接触到前面所有信息

---

## 七、特征融合（FPN / PAN）详解

### 7.1 自顶向下为什么从 P4 开始

以 `yolov8-p2.yaml` 为例：

Backbone 输出层级：
- P2（层 2）：160×160，128 通道（小目标）
- P3（层 4）：80×80，256 通道（中小目标）
- P4（层 6）：40×40，512 通道（中目标）
- P5（层 9/SPPF）：20×20，1024 通道（大目标）

Head 的 top-down 路径：
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 从 SPPF(P5) 上采样到 40×40
  - [[-1, 6], 1, Concat, [1]]                    # 和 backbone P4(层6) 融合
  - [-1, 3, C2f, [512]]                          # 得到融合后的 P4
```

**不是"从 P4 开始"，而是"从 P5 开始，第一步和 P4 融合"**。逻辑如下：

1. **起点是 P5**（SPPF 输出，20×20），因为它有最强的语义信息（大感受野）
2. 通过 `Upsample` 把 P5 分辨率提升到 40×40，和 backbone P4 对齐
3. 用 `Concat` 把两者通道拼起来，再用 `C2f` 融合
4. 接着把融合后的 P4 再上采样到 80×80，和 P3 融合……以此类推直到 P2

**为什么从 P5 往 P2 走，而不是反过来？**

- 深层（P5）语义强但位置粗，浅层（P2）位置准但语义弱
- 自顶向下（P5→P2）是把**强语义信息往浅层传递**，让小目标检测也能获得上下文理解
- 如果反过来（P2→P5），浅层的位置信息往深层传意义不大，因为深层本身感受野已经很大了

### 7.2 融合做了什么操作

以 `yolov8-p2.yaml` 中 P5→P4 融合为例：

#### Step 1: Upsample（分辨率对齐）
```yaml
- [-1, 1, nn.Upsample, [None, 2, "nearest"]]
```
- 输入：P5 特征图（假设 1024 通道，20×20）
- 操作：最近邻插值，分辨率翻倍 → 40×40
- 输出：1024 通道，40×40（通道数不变，空间尺寸翻倍）

#### Step 2: Concat（通道拼接）
```yaml
- [[-1, 6], 1, Concat, [1]]
```
- `[-1, 6]` 表示取两个输入：上一层输出（Upsample 后的 P5）和 backbone 第 6 层（P4）
- `Concat` 在通道维度拼接
- 输入1：1024 通道（来自 P5 Upsample）
- 输入2：512 通道（来自 backbone P4）
- 输出：1024 + 512 = **1536 通道**，40×40

#### Step 3: C2f（特征融合 + 降维）
```yaml
- [-1, 3, C2f, [512]]
```
- 输入：1536 通道（Concat 输出）
- `C2f` 内部先 `cv1` 用 1x1 卷积处理，再过 Bottleneck，最后 `cv2` 输出
- 输出：**512 通道**，40×40 —— 这就是融合后的新 P4 特征

所以一次完整的融合 = **分辨率对齐 → 通道拼接 → 深层融合降维**。

### 7.3 完整 FPN + PAN 流程图

```
Backbone                          Head (Neck)
P5 (20×20) ──► SPPF ───────────────────────┐
                                           │ Upsample ×2
                                           ▼
P4 (40×40) ───────────────────────┐    [P5_up] ──► Concat ──► C2f ──► 融合P4
                                  │                           │
                                  │                    Upsample ×2
                                  │                           ▼
P3 (80×80) ───────────────┐    [P4_up] ──────────────────► Concat ──► C2f ──► 融合P3
                          │                                 │
                          │                          Upsample ×2
                          │                                 ▼
P2 (160×160) ────────┐ [P3_up] ─────────────────────────► Concat ──► C2f ──► 融合P2
                     │                                      │
                     │                                      │ Conv 3×2 stride=2
                     │                                      ▼
                     │                                [P2_down] ──► Concat ──► C2f ──► buP3
                     │                                              │
                     │                                              │ Conv 3×2
                     │                                              ▼
                     │                                        [buP3_down] ──► Concat ──► C2f ──► buP4
                     │                                                    │
                     │                                                    │ Conv 3×2
                     │                                                    ▼
                     │                                              [buP4_down] ──► Concat ──► C2f ──► buP5
                     │
                     └──────────────────────────────────────────────────────────────────────────────┘
                                                                                │
                                                                                ▼
                                                                    Detect(P2, P3, P4, P5)
```

- **左侧 Backbone**：逐级下采样，提取从细到粗的特征
- **FPN top-down（上采样路径）**：把深层语义往浅层传，解决小目标语义不足
- **PAN bottom-up（下采样路径）**：把浅层精确的位置信息重新往深层传，解决定位不准
- **Detect**：在 P2/P3/P4/P5 四个尺度上并行预测，小目标用 P2，大目标用 P5

---

## 八、快速速查表

| 我想做... | 操作位置 |
|-----------|----------|
| 加/删 CBAM 层 | 编辑 `cfg/models/v8/*.yaml` |
| 加 P2 检测头 | 修改 head：添加 Upsample→Concat→C2f，Detect 改 4 输入 |
| 加全新的模块类 | `nn/modules/conv.py` → `__init__.py` → `tasks.py` |
| 改类别数 | YAML 顶部 `nc: 6` |
| 改模型缩放比例 | YAML `scales:` 或加载时选择 `n/s/m/l/x` |
| 改训练参数 | Python 脚本里 `model.train(...)` |
| 改特征融合结构 | 编辑 head 里的 Upsample/Concat/C2f 顺序和连接关系 |
