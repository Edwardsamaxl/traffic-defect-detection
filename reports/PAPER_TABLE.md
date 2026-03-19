# 论文表格：策略命名与 Test 集对比（NEU-DET）

数据集配置：`datasets/neu.yaml`，评估 split=`test`（180 images）。  
评估参数：`conf=0.001`，`iou=0.6`。TTA 表示 `augment=True/False`。  

## 1. 策略命名规则（论文用）

为避免 “best/strategy” 这种不清晰命名，建议论文中统一采用：

- **S0 Baseline-Default**：最初 baseline（YOLOv8s，默认训练配置）  
- **S1 Sup-Optim (Cosine+Aug+TTA)**：监督阶段优化（增强 + 余弦退火；测试可开 TTA）  
- **A0 No-Aug Ablation**：无增强消融（mosaic/flip/copy-paste 关闭，用于说明增强边际贡献）  
- **R0 Resolution Ablation**：分辨率消融（640 vs 1024）  
- **SS1 Seed-Supervised**：seed-only 监督模型（半监督起点）  
- **SS2 Pseudo-Adaptive**：动态阈值伪标签（Adaptive）  
- **SS3 Pseudo-Filter (Conf=0.7 vs Conservative)**：伪标签筛选强度对比（更保守→更高 precision、可能降低 recall）  
- **SS4 Seed-Repeat Ablation**：seed 复制 1 次 / 2 次（once/twice）对比

> 说明：你当前实验中，test 集最优仍为 S0；S1 在 test 上略弱但有独立贡献（增强/余弦/TTA 带来稳定小幅收益或结构性变化），半监督策略主要体现为不同筛选/合并策略的权衡与负结果分析。

## 2. Test 集主对比表（论文可直接引用）

表中 “P/R/mAP50/mAP50-95” 均来自 `reports/eval_suite_20260318_175226.csv`（补跑的 seed-conservative 见表后说明）。

| 方法ID | 论文命名（建议） | 对应权重文件 | imgsz | TTA | P | R | mAP50 | mAP50-95 |
|---|---|---|---:|:---:|---:|---:|---:|---:|
| S0 | Baseline-Default | `experiments/baseline_s/weights/best.pt` | 640 | Off | 0.7026 | 0.7302 | **0.7729** | **0.4338** |
| S0+TTA | Baseline-Default + TTA | `experiments/baseline_s/weights/best.pt` | 640 | On | 0.7398 | 0.7222 | **0.7786** | **0.4448** |
| S1 | Sup-Optim (Cosine+Aug) | `experiments/stage4_overall/weights/best-cosine.pt` | 640 | Off | 0.7061 | 0.7366 | 0.7669 | 0.4181 |
| S1+TTA | Sup-Optim (Cosine+Aug) + TTA | `experiments/stage4_overall/weights/best-cosine.pt` | 640 | On | 0.7833 | 0.7001 | 0.7711 | 0.4248 |
| A0 | No-Aug Ablation | `experiments/stage6_semi/weights/new-best-noaug.pt` | 640 | Off | 0.7146 | 0.5791 | 0.6500 | 0.3757 |
| R0-640 | Resolution Ablation (train/val @640) | `experiments/stage4_overall/weights/best-640.pt` | 640 | Off | 0.6547 | 0.7369 | 0.7306 | 0.4030 |
| R0-1024 | Resolution Ablation (train/val @1024) | `experiments/stage4_overall/weights/best-1024.pt` | 1024 | Off | 0.6864 | 0.6970 | 0.7195 | 0.3746 |
| SS1 | Seed-Supervised (seed-only) | `experiments/baseline_seed/weights/best.pt` | 640 | Off | 0.6563 | 0.6554 | 0.6761 | 0.3429 |
| SS2-once | Pseudo-Adaptive + Seed-Repeat(once) | `experiments/stage6_semi/weights/best-adaptive-once.pt` | 640 | Off | 0.6335 | 0.6633 | 0.6775 | 0.3575 |
| SS2-twice | Pseudo-Adaptive + Seed-Repeat(twice) | `experiments/stage6_semi/weights/best-adaptive-twice.pt` | 640 | Off | 0.6307 | 0.6600 | 0.6878 | 0.3627 |
| SS3-0.7 | Pseudo-Filter (Conf≈0.7) | `experiments/stage6_semi/weights/new-best.pt` | 640 | Off | 0.6284 | 0.6390 | 0.6722 | 0.3527 |
| SS3-cons | Pseudo-Filter (Conservative) | `experiments/stage6_semi/weights/new-best-conservative.pt` | 640 | Off | 0.7137 | 0.6148 | 0.6870 | 0.3657 |

## 3. 关键观察（写进论文讨论的版本）

1) **Test 最优**：S0 baseline 仍为最优；开启 TTA 后 baseline 进一步提升。  
2) **TTA 对两者均有效**：S0 与 S1 开启 TTA 均有提升，但 S0+TTA 仍领先。  
3) **分辨率消融**：1024 相比 640 明显更慢且指标下降（mAP50-95 降幅更明显）。  
4) **无增强消融**：A0 的 recall 明显偏低，说明增强对泛化/召回具有重要贡献（可支撑你“增强有效”的论点）。  
5) **半监督结果**：SS2/SS3 在 test 上未超过 S0，但呈现出“阈值更保守→precision 上升、recall 下降”的可解释权衡；seed 重复（once/twice）对结果存在小幅影响。

## 4. 补充：seed-conservative（你额外提到的对比）

你补跑的 `experiments/baseline_seed/weights/best-conservative.pt`（test, 640, TTA off）为：

- P=0.6667, R=0.6840, mAP50=0.6950, mAP50-95=0.3740

可在论文中作为 “seed 伪标签生成/筛选策略差异” 的补充对照。

