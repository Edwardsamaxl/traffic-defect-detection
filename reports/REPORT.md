# Traffic Defect Detection 实验阶段报告

## 1. 当前阶段结论

本项目已经完成从训练脚本到可演示系统的闭环，包含：

- 基线训练与多轮半监督策略实验
- 伪标签生成、数据合并、离线增强流程
- Ultralytics 本地源码改造（伪标签损失加权）
- 可运行的 Web 前后端（单图展示 + 批量文件夹打标 + 模型切换）

当前最重要结论：半监督性能提升已接近 seed 能力上限，后续价值更高的工作是工程落地与实验总结，而不是盲目继续调参。

## 2. 已做策略试验清单

### 2.1 监督与数据增强基线

1) Baseline 监督训练  
2) Offline Copy-Paste 增强（离线生成增强样本）  
3) 基于 seed 的监督训练（作为半监督起点）

### 2.2 伪标签策略

1) Conservative 伪标签阈值策略  
2) Adaptive 动态阈值伪标签策略  
3) Flip-consistency（翻转一致性）筛选伪标签  
4) 伪标签与 seed 的多种合并策略（train-only / adaptive / conservative）

### 2.3 半监督训练策略

1) Stage6 半监督训练（基于 merge 数据）  
2) Teacher-Student 迭代方案尝试（含 Kaggle 适配验证）  
3) 单轮方案回退：使用离线 merge 数据直接训练（不再迭代重打标）

### 2.4 损失层改造

1) 在 Ultralytics 源码中加入 `pseudo_weight` / `pseudo_key`  
2) 让伪标签样本损失可做权重衰减（而非仅靠采样比例）

## 3. 关键实验观察（可写入论文讨论）

1) 动态阈值策略相比固定阈值更稳，但收益逐步递减。  
2) Teacher-Student 迭代在当前数据条件下未明显优于单轮高质量伪标签方案。  
3) 主要瓶颈来自 seed 上限与伪标签噪声传播，典型弱类提升有限。  
4) 继续“纯调参”边际收益较低，工程化闭环与误差分析更具毕业设计价值。

## 4. 代表性结果快照（当前可复述）

以最新单次评估结果为例（`src/utils/evaluation.py`）：

- Precision: 0.6861  
- Recall: 0.6574  
- mAP@0.5: 0.7049  
- mAP@0.5:0.95: 0.3791

类别层面：`patches`、`scratches` 表现较好；`crazing`、`rolled-in_scale` 仍是主要短板类别。

## 5. 当前工程化状态

`src/webapp` 已支持：

1) 单图检测与可视化展示  
2) 批量文件夹检测  
3) 项目内模型自动扫描 + 本地 `.pt` 上传  
4) 阈值预设（平衡 / 高召回 / 高精度 / 自定义）  
5) 批量结果自动落盘到 `output/webapp_batch_outputs/<run_id>/`，并生成 `summary.json`

## 6. 论文可写结构建议

建议优先写下面 4 块（和你已有结果最匹配）：

1) **方法部分**：从 seed-supervised 到 pseudo-labeling，再到 loss weighting。  
2) **实验部分**：baseline、adaptive、consistency、weighted、offline-merge 对比。  
3) **消融部分**：是否使用动态阈值、是否一致性筛选、是否加权。  
4) **分析部分**：按类别误差与失败案例，解释“提升趋缓”的原因。

## 7. 下一步建议（执行优先级）

1) 固化一版对比表（每类 mAP50 + 总 mAP）。  
2) 从 `output/webapp_batch_outputs` 选取 20~30 张失败样例做误差归因。  
3) 把实验流程图和系统架构图补齐（训练侧 + 部署侧）。  
4) 再决定最终论文保留哪些策略，避免叙事过散。
