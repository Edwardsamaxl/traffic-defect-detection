# 半监督学习方案详解

## 目录

1. [已实现的半监督学习方案](#1-已实现的半监督学习方案)
2. [动态置信度阈值深入分析](#2-动态置信度阈值深入分析)
3. [翻转一致性正则深入分析](#3-翻转一致性正则深入分析)
4. [Seed数据强化方案](#4-seed数据强化方案)
5. [后续发展计划](#5-后续发展计划)

---

## 1. 已实现的半监督学习方案

### 1.1 整体框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          半监督学习框架                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐  │
│   │  标注数据     │    │   无标注数据      │    │     伪标签生成        │  │
│   │  (378张)     │    │   (882张)        │    │   (动态阈值+一致性)    │  │
│   └──────┬───────┘    └────────┬────────┘    └──────────┬───────────┘  │
│          │                      │                       │               │
│          │                      │                       ▼               │
│          │                      │              ┌──────────────────┐      │
│          │                      │              │   伪标签数据      │      │
│          │                      │              │   (高质量筛选)    │      │
│          │                      │              └────────┬─────────┘      │
│          │                      │                       │                  │
│          ▼                      ▼                       ▼                  │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                     混合训练数据集                          │          │
│   │              (标注数据 + 伪标签数据 + Seed强化)                │          │
│   └─────────────────────────────┬───────────────────────────────┘          │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                     YOLOv8s 模型训练                        │            │
│   └─────────────────────────────────────────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 伪标签生成策略

| 方法 | 类 | 说明 |
|------|-----|------|
| `generate_standard` | PseudoLabelGenerator | 固定置信度阈值（如0.7） |
| `generate_adaptive` | PseudoLabelGenerator | 基于类别AP的动态阈值 |
| `generate_consistency` | PseudoLabelGenerator | 翻转一致性筛选 |
| `generate_adaptive_consistency` | PseudoLabelGenerator | **组合方法**：动态阈值 + 翻转一致性 |
| `generate_with_uncertainty` | PseudoLabelGenerator | 基于MC-Dropout的不确定性估计 |

### 1.3 半监督训练器

| 类 | 文件 | 核心思想 |
|----|------|----------|
| `FixMatchTrainer` | fixmatch.py | 弱增强伪标签 + 强增强一致性 |
| `CurriculumPseudoLabelTrainer` | fixmatch.py | 课程学习：由高到低逐步降低阈值 |
| `NoisyStudentTrainer` | fixmatch.py | Teacher-Student迭代训练 |

---

## 2. 动态置信度阈值深入分析

### 2.1 核心思想

**问题**：固定阈值对所有类别一刀切，但不同类别的检测难度差异很大。

- AP高的类别（如scratches 0.75）→ 检测容易 → 应使用较低阈值
- AP低的类别（如crazing 0.40）→ 检测困难 → 应使用较高阈值

### 2.2 自适应阈值公式

**第一步**：训练监督学习基线，获取各类别AP

$$
\text{AP}_c = \text{类别}c\text{的Average Precision (mAP@IoU=0.5)}
$$

**第二步**：归一化AP到[0,1]

$$
\text{norm\_AP}_c = \frac{\text{AP}_c - \min(\text{AP})}{\max(\text{AP}) - \min(\text{AP}) + \epsilon}
$$

**第三步**：计算自适应阈值

$$
\text{threshold}_c = \text{base\_conf} + \lambda \cdot (1 - \text{norm\_AP}_c)
$$

### 2.3 配置参数

```python
# src/utils/config.py 中的默认值
PSEUDO_LABEL_DEFAULTS = {
    "base_conf": 0.65,          # 基础置信度
    "adaptive_lambda": 0.25,    # 调节力度
    "iou_match": 0.6,           # IoU匹配阈值
    "standard_conf": 0.7,       # 标准伪标签阈值
}
```

### 2.4 阈值计算示例

| 类别 | AP | norm_AP | threshold (base=0.65, λ=0.25) |
|------|-----|---------|------------------------------|
| scratches | 0.75 | 1.00 | 0.65 |
| inclusion | 0.60 | 0.50 | 0.775 |
| patches | 0.55 | 0.40 | 0.85 |
| rolled-in_scale | 0.50 | 0.30 | 0.925 |
| pitted_surface | 0.45 | 0.15 | 1.00 (clamped) |
| crazing | 0.40 | 0.00 | 0.95 |

**归一化细节**：
- AP范围: [0.40, 0.75]
- AP_min = 0.40, AP_max = 0.75
- 以crazing (AP=0.40)为例: norm = (0.40 - 0.40) / (0.75 - 0.40) = 0.00
- 以scratches (AP=0.75)为例: norm = (0.75 - 0.40) / (0.75 - 0.40) = 1.00

### 2.5 核心代码

```python
def _compute_adaptive_threshold(
    self,
    class_id: int,
    base_conf: float = 0.65,
    lambda_val: float = 0.25,
) -> float:
    """
    计算自适应阈值

    AP 低的类别使用更高阈值，AP 高的类别使用更低阈值
    """
    ap = self.baseline_ap[class_id]
    ap_min = min(self.baseline_ap)
    ap_max = max(self.baseline_ap)

    # 归一化
    norm = (ap - ap_min) / (ap_max - ap_min + 1e-9)

    # 阈值 = base_conf + lambda * (1 - norm)
    threshold = base_conf + lambda_val * (1 - norm)
    return max(0.3, min(0.95, threshold))  # 限制在 [0.3, 0.95]
```

---

## 3. 翻转一致性正则深入分析

### 3.1 核心思想

通过**水平翻转**构造一致性正则：同一图像的翻转版本应该给出相同的检测结果。如果原图和翻转图的预测不一致，说明该检测结果不可靠。

### 3.2 流程图

```
原图预测                    翻转图预测
    │                          │
    ▼                          ▼
┌─────────┐                ┌─────────┐
│ Boxes   │                │ Boxes   │
└────┬────┘                └────┬────┘
     │                          │
     │                     (x' = 1 - x)
     │                          │
     ▼                          ▼
┌─────────────────────────────────────┐
│       IoU匹配 + 类别一致性            │
│   IoU(box, flip_box) >= τ=0.6       │
└────────────────┬────────────────────┘
                 │
                 ▼
         保留伪标签 + 取min(conf)
```

### 3.3 算法步骤

1. **原图预测**：对原始图像进行目标检测，得到 `(class_id, conf, xywh)`
2. **翻转预测**：对水平翻转图像进行检测
3. **坐标映射**：将翻转图的边界框x坐标映射回原图坐标
   - 由于翻转后水平镜像：$x_{mapped} = 1.0 - x_{flip}$
4. **IoU匹配**：对于每个原图检测框，找到翻转图中**相同类别**的最佳匹配
5. **一致性验证**：仅保留 IoU ≥ τ (0.6) 的检测框
6. **置信度融合**：使用 $\min(conf_{orig}, conf_{flip})$ 作为最终置信度

### 3.4 为什么要取min置信度？

取min是一种**悲观融合**策略：
- 原图置信度 0.9，翻转图置信度 0.5 → 取0.5
- 如果两个视图都高度确信某个目标，才给予高置信度
- 有效降低假阳性率

### 3.5 核心代码

```python
def generate_consistency(
    self,
    unlabeled_dir: Path | str,
    output_dir: Path | str,
    base_conf: float = 0.65,
    iou_match: float = 0.6,
):
    """翻转一致性伪标签生成"""
    # ...
    for img_path in image_paths:
        img = cv2.imread(str(img_path))

        # 原图预测
        res_orig = self.model.predict(source=img, augment=False)[0]

        # 翻转图预测
        res_flip = self.model.predict(source=cv2.flip(img, 1), augment=False)[0]

        # 收集原图 boxes
        orig_boxes = []
        for b in res_orig.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            if conf >= base_conf:
                xywh = b.xywhn[0].tolist()
                orig_boxes.append((cid, conf, xywh))

        # 收集翻转 boxes (x坐标翻转)
        flip_boxes = []
        for b in res_flip.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            if conf >= base_conf:
                x, y, w, h = b.xywhn[0].tolist()
                x = 1.0 - x  # 翻转 x 坐标
                flip_boxes.append((cid, conf, [x, y, w, h]))

        # 一致性匹配
        valid_boxes = []
        for cid, conf, xywh in orig_boxes:
            box_xyxy = self._yolo_to_xyxy(xywh)
            matched = False

            for fcid, fconf, fxywh in flip_boxes:
                if fcid != cid:
                    continue
                if self._iou_xyxy(box_xyxy, self._yolo_to_xyxy(fxywh)) >= iou_match:
                    matched = True
                    # 使用较低置信度
                    final_conf = min(conf, fconf)
                    break

            if matched:
                valid_boxes.append((cid, xywh[0], xywh[1], xywh[2], xywh[3]))
```

---

## 4. Seed数据强化方案

### 4.1 动机

伪标签虽然经过动态阈值+一致性筛选，但质量仍不如真实标注。需要**提升标注数据的相对权重**，降低伪标签的相对影响。

### 4.2 策略

| 数据类型 | 复制次数 | 相对权重 |
|---------|---------|---------|
| Seed (标注数据) | 3x | 1.0 |
| 伪标签数据 | 1x | 0.5 (有效) |

### 4.3 效果分析

```
标注数据:  378张 × 3复制 = 1134个训练样本（每epoch出现3次）
伪标签数据: 882张 × 1 = 882个训练样本（每epoch出现1次）

有效权重比:
- Seed: 1134 × 1.0 = 1134
- Pseudo: 882 × 0.5 = 441

等效伪标签权重 ≈ 441 / (1134 + 441) ≈ 0.28
```

### 4.4 实现方式

在数据集yaml中配置：
```yaml
# neu_merge.yaml
train:
  - images/train      # Seed数据 (378张)
  - images/train_seed # Seed数据复制 (×3)
  - images/unlabeled  # 伪标签数据 (882张)
```

---

## 5. 后续发展计划

### 5.1 迭代伪标签（当前只做了一轮）

**当前方案**（单轮）：
```
Baseline训练 → 生成伪标签 → 混合训练 → 最终模型
```

**迭代伪标签**（多轮）：
```
Baseline训练
      │
      ▼
第一轮伪标签 → 混合训练 → 模型1
      │                    │
      ▼                    ▼
      └─── 模型1 → 第二轮伪标签 → 混合训练 → 模型2
                                            │
                                            ▼
                                   更高质量伪标签 + 更强模型
```

**改进方向**：
- 每轮使用当前最佳模型重新生成伪标签
- 逐步降低置信度阈值，放宽筛选条件
- 预期收益：挖掘更多困难样本

### 5.2 课程学习伪标签训练流程

```python
class CurriculumPseudoLabelTrainer:
    def train(self, stages):
        """
        分阶段训练，难度递增
        """
        stages = [
            {"conf_threshold": 0.9, "epochs": 50},   # 阶段1: 只用easy
            {"conf_threshold": 0.8, "epochs": 50},   # 阶段2: 加入medium
            {"conf_threshold": 0.7, "epochs": 100},  # 阶段3: 加入hard
        ]

        for stage_idx, stage in enumerate(stages):
            # 用上一阶段的模型初始化
            # 学习率逐渐降低
            lr0 = 0.001 * (0.9 ** stage_idx)

            # 训练
            self.model.train(epochs=stage["epochs"], conf_threshold=stage["conf_threshold"])
```

**核心思想**：
- 先学习简单样本，建立稳固的决策边界
- 逐步加入困难样本，避免被噪声干扰
- 符合人类学习由浅入深的规律

### 5.3 Noisy Student训练流程

```python
class NoisyStudentTrainer:
    def train(self):
        """
        Teacher-Student迭代训练
        """
        # Step 1: 用小模型(如YOLOv8s)作为Teacher
        teacher = YOLO("yolov8s.pt")

        # Step 2: Teacher生成伪标签
        pseudo_labels = generate_pseudo_labels(teacher, unlabeled_data)

        # Step 3: 用大模型(如YOLOv8m)作为Student在混合数据上训练
        student = YOLO("yolov8m.pt")
        student.train(data="labeled + pseudo_labels")

        # Step 4: Student成为新的Teacher，迭代
```

**关键要素**：
- **模型规模**：Student比Teacher大（s→m→l）
- **数据噪声**：Student训练时使用更强的数据增强
- **迭代**：每轮用更大的模型 + 更高质量的伪标签

### 5.4 可能的改进方向

#### 5.4.1 不确定性估计

**方法**：使用MC-Dropout进行多次推理，估算预测不确定性

```
同一图像 → T次随机Dropout推理 →
    ├── 预测框位置方差 → 低方差=可靠
    └── 预测类别熵 → 低熵=可靠
```

**代码已在`generate_with_uncertainty`中实现**（见pseudo_label_generator.py:489-577），使用多次推理选择最稳定的预测。

#### 5.4.2 对比学习

**思想**：让同一目标的在不同增强视图下的特征表示更接近

```
增强视图1 (原图)     增强视图2 (翻转)
      │                    │
      ▼                    ▼
   特征f1               特征f2
      │                    │
      └────────┬───────────┘
               ▼
         对比损失: ||f1 - f2||²
```

**应用于检测**：
- 正样本对：同一目标的原图/翻转/裁剪特征
- 负样本对：不同目标的特征
- 目标：提升特征判别性，改善小目标检测

#### 5.4.3 软标签与硬标签混合

| 类型 | 标签形式 | 损失函数 |
|------|---------|---------|
| 硬标签 | one-hot | CrossEntropy |
| 软标签 | 概率分布 | KL散度 |

```
teacher预测: [0.1, 0.7, 0.2]  → 软标签
硬标签:     [0,   1,   0  ]

损失 = α × CE(硬标签) + β × KL(软标签 || 预测)
```

**优势**：软标签包含更多信息（类别间相似度），比硬标签更平滑

#### 5.4.4 MixUp/CutMix 半监督

```python
# MixUp 半监督
def mixup_label(labeled_y, pseudo_y, lambda_mix):
    """
    混合标注标签和伪标签
    """
    # 边界框混合
    box_mixed = lambda_mix * box_labeled + (1 - lambda_mix) * box_pseudo

    # 类别标签混合（软标签）
    cls_mixed = lambda_mix * one_hot(labeled) + (1 - lambda_mix) * pseudo_probs

    return box_mixed, cls_mixed
```

**优势**：
- 更多样化的训练样本
- 正则化效果，防止过拟合

---

## 面试要点总结

### Q1: 为什么需要动态置信度阈值？

**答**：不同类别的检测难度差异大。固定阈值会导致：
- AP高的类别（容易检测）阈值过高 → 漏掉很多正确预测
- AP低的类别（难以检测）阈值过低 → 引入大量假阳性

动态阈值根据类别AP自适应调整：AP高→低阈值，AP低→高阈值。

### Q2: 翻转一致性的核心思想是什么？

**答**：基于**视角不变性**假设：水平翻转不应改变检测结果。
- 原图和翻转图都应该检测到相同的目标
- 如果不一致，说明至少有一个是假阳性
- 通过IoU匹配筛选出跨视角一致的预测

### Q3: 为什么取min而不是平均融合置信度？

**答**：取min是**悲观策略**，只有两个视图都确信才给高置信度。适用于：
- 伪标签质量参差不齐
- 需要严格控制假阳性率
- 宁可少标注，不要错误标注

### Q4: Seed数据强化的原理？

**答**：通过数据复制提升标注数据的有效权重：
- 伪标签质量 ≈ 0.5倍真实标签
- 通过3x复制，伪标签的相对权重降低到 ~0.17
- 防止模型过度依赖伪标签的噪声

### Q5: 迭代伪标签相比单轮有什么优势？

**答**：
- 每轮用更强的模型生成更高质量的伪标签
- 可以逐步降低阈值，挖掘更多困难样本
- 形成正循环：更强模型 → 更好伪标签 → 更强模型
