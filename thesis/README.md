# 钢材表面缺陷检测系统 - 论文文档

## 项目概述

基于 YOLOv8 的钢材表面缺陷目标检测，采用半监督学习策略减少标注需求。

**数据集**: NEU-DET（6类缺陷：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches）

## 论文结构

### 第一章 绪论
- 研究背景：工业质检、钢材缺陷检测的重要性
- 研究目标：利用半监督学习减少标注需求
- 贡献总结（3-4点）

### 第二章 相关技术与文献综述
- 2.1 目标检测发展（两阶段→单阶段→Transformer）
- 2.2 YOLO系列演进（重点YOLOv8）
- 2.3 半监督学习（FixMatch、Noisy Student等）
- 2.4 小样本/半监督目标检测现状

### 第三章 数据集与问题定义
- 3.1 NEU-DET数据集介绍（6类缺陷、可视化示例）
- 3.2 数据分布分析（各类别数量不平衡）
- 3.3 半监督场景定义（labeled/unlabeled划分）
- 3.4 评价指标（mAP@50/75, Precision, Recall）

### 第四章 方法论（核心章节）
- 4.1 整体框架图
- 4.2 基线模型：YOLOv8s 监督学习
- 4.3 半监督策略
  - **动态置信度阈值**（Class-wise adaptive threshold based on AP）
  - **翻转一致性筛选**（Flip consistency filtering）
  - **Seed数据强化**（Seed数据复制降低伪标签权重）
- 4.4 消融实验：验证每个模块的贡献

### 第五章 实验结果
- 5.1 实验设置（训练配置、硬件环境）
- 5.2 监督学习基线对比
- 5.3 半监督学习效果
- 5.4 消融实验（各策略有效性）
- 5.5 与现有方法对比
- 5.6 可视化分析（检测结果示例）

### 第六章 系统展示（Demo章节）
- 6.1 Web API 设计
- 6.2 单图检测演示
- 6.3 批量检测演示
- 6.4 实际应用场景

### 第七章 总结与展望
- 工作总结
- 局限性讨论
- 未来方向

## 文件结构

```
thesis/
├── README.md                    # 本文件
├── structure.md                 # 详细章节结构
├── figures/                     # 图表文件
│   ├── framework.png           # 技术框架图
│   ├── dataset_samples.png     # 数据集示例
│   ├── data_distribution.png   # 数据分布图
│   ├── adaptive_threshold.png   # 动态阈值流程图
│   ├── flip_consistency.png     # 翻转一致性示意图
│   ├── training_curves/        # 训练曲线
│   └── detection_results/      # 检测结果对比
├── tables/                      # 表格数据
│   ├── supervised_baseline.csv # 监督学习基线
│   ├── semi_supervised.csv     # 半监督实验结果
│   ├── ablation.csv            # 消融实验
│   └── comparison_sota.csv      # 与SOTA对比
└── scripts/                     # 辅助脚本
    ├── visualize_dataset.py    # 数据集可视化
    ├── plot_training_curves.py # 训练曲线绘制
    └── generate_results_table.py # 结果表格生成
```

## 关键创新点

1. **动态置信度选择**: 基于类别AP的自适应阈值，而非固定阈值
2. **Seed强化策略**: 通过数据复制模拟伪标签权重降低
3. **翻转一致性筛选**: 提升伪标签质量

## 实验状态

| 实验 | 状态 | 说明 |
|------|------|------|
| 监督学习基线 | ✅ 完成 | YOLOv8s |
| 半监督（自适应阈值） | ✅ 有效 | 动态置信度 + 翻转一致性 |
| Demo API | ✅ 可用 | FastAPI 单图/批量检测 |

## 待完成

- [ ] Kaggle训练完成
- [ ] 整理所有实验的mAP、Precision、Recall表格
- [ ] 消融实验数据补全
- [ ] 数据集可视化素材
- [ ] 训练曲线收集
- [ ] 检测结果对比图
- [ ] 混淆矩阵
