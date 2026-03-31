# 论文详细结构

## 第一章 绪论

### 1.1 研究背景与意义
- 工业质检的重要性
- 钢材表面缺陷检测的挑战
  - 人工检测效率低、成本高
  - 缺陷种类多、形态复杂
  - 标注数据获取困难
- 自动化检测的发展趋势

### 1.2 研究目标
- 利用深度学习实现自动化缺陷检测
- 采用半监督学习策略减少标注需求
- 构建完整的检测系统

### 1.3 主要贡献
1. 提出基于类别AP的动态置信度阈值策略
2. 设计翻转一致性筛选提升伪标签质量
3. 构建端到端的半监督目标检测框架
4. 开发可视化的检测Demo系统

### 1.4 论文结构

---

## 第二章 相关技术与文献综述

### 2.1 目标检测技术发展
- 两阶段检测器（R-CNN系列）
  - R-CNN
  - Fast R-CNN
  - Faster R-CNN
- 单阶段检测器（YOLO系列）
  - YOLOv1-v3：基本架构确立
  - YOLOv4-v5：工程优化
  - YOLOv8：Ultralytics版本
- Transformer架构（DETR系列）
- 各架构对比分析

### 2.2 YOLO系列详细演进
- YOLOv1：端到端单阶段检测的开创者
- YOLOv2-v3：性能持续提升
- YOLOv4：Bag of Freebies策略
- YOLOv5：工程化与易用性
- YOLOv8：最新架构特点
  - C2f模块
  - Anchor-free设计
  - 更好的精度/速度平衡

### 2.3 半监督学习技术
- 半监督学习基本概念
  - 一致性正则化
  - 伪标签技术
  - 协同训练
- 经典半监督方法
  - FixMatch：弱增强+强增强
  - Noisy Student：知识蒸馏
  - MixMatch：混合增强
- 半监督目标检测发展
  - CSD (Consistency-based Semi-supervised Detection)
  - STAC (Self-Training and Contrastive)
  - 现有方法的局限性分析

### 2.4 小样本/半监督目标检测现状
- Few-shot检测
- 半监督目标检测数据集
- 评估协议与指标

---

## 第三章 数据集与问题定义

### 3.1 NEU-DET数据集介绍
- 数据集来源与背景
- 6类缺陷详细说明
  - Crazing（裂纹）：细线状裂纹
  - Inclusion（夹杂）：金属中异物
  - Patches（斑块）：表面颜色异常
  - Pitted Surface（点蚀）：小坑状缺陷
  - Rolled-in Scale（氧化皮）：高温氧化层
  - Scratches（划痕）：线性擦伤
- 数据集规模与划分

### 3.2 数据分布分析
- 类别不平衡问题
  - 各类别样本数量统计
  - 标注边界框数量分布
- 缺陷尺寸分布
- 缺陷位置分布

### 3.3 半监督场景定义
- 数据划分策略
  - Labeled set: 378 images (30%)
  - Unlabeled set: 882 images (70%)
- 问题形式化
- 训练流程

### 3.4 评价指标
- 目标检测常用指标
  - IoU (Intersection over Union)
  - Precision & Recall
  - AP (Average Precision)
  - mAP (mean Average Precision)
- 本文采用的指标
  - mAP@0.5：标准阈值
  - mAP@0.5:0.95：严格评估
  - Precision/Recall：实际应用参考

---

## 第四章 方法论（核心章节）

### 4.1 系统框架概述
- 整体流程图
- 三大核心模块
  1. 基线模型（YOLOv8s）
  2. 伪标签生成（动态阈值+一致性筛选）
  3. 半监督训练（标签数据+伪标签数据）

### 4.2 基线模型：YOLOv8s
- 模型架构
  - Backbone: CSPDarknet
  - Neck: PAFPN
  - Head: 解耦检测头
- 损失函数
  - Bbox Loss: CIoU
  - Class Loss: Focal Loss
  - DFL Loss: Distribution Focal Loss
- 训练配置

### 4.3 半监督策略（核心创新）

#### 4.3.1 动态置信度阈值
**动机**：固定阈值对不同类别效果不均

**方法**：
1. 首先训练监督学习基线，获取各类别AP
2. 计算自适应阈值：
   ```
   threshold[c] = base_conf + λ * (1 - norm_AP[c])
   ```
3. AP低的类别使用更高阈值，减少伪标签噪声
4. AP高的类别使用较低阈值，保留更多正样本

**伪代码**：
```python
def compute_adaptive_threshold(baseline_ap, class_id):
    ap = baseline_ap[class_id]
    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)
    norm = (ap - ap_min) / (ap_max - ap_min + ε)
    return base_conf + λ * (1 - norm)
```

#### 4.3.2 翻转一致性筛选
**动机**：单次预测不稳定，需要验证预测一致性

**方法**：
1. 对原始图像进行水平翻转
2. 分别对原图和翻转图进行预测
3. 将翻转图的预测坐标映射回原图
4. 只保留IoU匹配且类别一致的预测
5. 使用较低置信度作为最终置信度

**公式**：
```
if IoU(box_orig, flip_box_mapped) >= τ:
    accept pseudo-label
```

#### 4.3.3 Seed数据强化
**动机**：伪标签质量不如真实标签，需要平衡

**方法**：
1. 将Seed数据复制多份
2. 降低伪标签样本的权重
3. 等效于减少伪标签的相对影响力

**实现**：
- Seed数据复制3次
- 伪标签数据权重设为0.5

### 4.4 消融实验设计
- 实验1：基线（无半监督）
- 实验2：标准伪标签（固定阈值）
- 实验3：自适应阈值
- 实验4：翻转一致性筛选
- 实验5：完整方法（自适应+一致性）

---

## 第五章 实验结果

### 5.1 实验设置
- 硬件环境
  - GPU: NVIDIA RTX 3080 / Kaggle P100
  - 内存: 16GB
- 软件环境
  - Python 3.9+
  - PyTorch 2.0+
  - Ultralytics YOLOv8
- 训练配置
  - Epochs: 200
  - Batch Size: 4
  - Image Size: 640
  - Optimizer: AdamW

### 5.2 监督学习基线
- YOLOv8s在NEU-DET上的性能
- 各类别AP分析
- 训练曲线

### 5.3 半监督学习效果
- 与监督学习对比
- 不同伪标签策略对比
- 各类别性能提升

### 5.4 消融实验结果
- 各组件贡献分析
- 动态阈值的效果
- 一致性筛选的效果

### 5.5 与现有方法对比
- 与FixMatch对比
- 与Noisy Student对比
- 与其他半监督方法对比

### 5.6 可视化分析
- 检测结果示例
- 成功案例
- 失败案例分析

---

## 第六章 系统展示

### 6.1 Web API设计
- FastAPI架构
- 端点设计
  - `/predict`: 单图检测
  - `/predict_batch`: 批量检测
  - `/models`: 模型管理

### 6.2 单图检测演示
- API调用示例
- 响应格式说明
- 可视化结果

### 6.3 批量检测演示
- 批量处理流程
- 结果汇总
- 输出格式

### 6.4 实际应用场景
- 工业生产线集成
- 质检流程自动化
- 边缘部署考虑

---

## 第七章 总结与展望

### 7.1 工作总结
- 主要贡献回顾
- 方法有效性验证
- 系统完成情况

### 7.2 局限性讨论
- 数据集规模限制
- 类别不平衡挑战
- 伪标签噪声累积

### 7.3 未来方向
- 更多缺陷类别
- 主动学习结合
- 端到端优化
- 实际部署优化

---

## 参考文献

## 附录
- A. 完整实验配置
- B. 额外检测结果
- C. 代码说明
