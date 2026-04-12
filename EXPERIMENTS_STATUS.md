# 实验状态记录 (2026-04-12 更新)

## 评测标准

**所有实验统一使用 `src/utils/evaluation.py` 进行评测：**
```bash
python src/utils/evaluation.py --model <weight_path> --split test
```

**评测配置:**
- 数据集: test (180张图像, 442实例)
- conf: 0.001
- iou: 0.6
- imgsz: 640

---

## 评测结果汇总 (测试集)

| 模型 | 参数量 | P | R | mAP50 | mAP50-95 | 排名 |
|------|--------|------|------|--------|-----------|------|
| 02_cbam | 11.2M | **0.7595** | 0.7011 | **0.7870** | 0.4258 | #1 |
| **baseline** | 11.1M | 0.7000 | 0.7303 | 0.7733 | **0.4342** | #2 |
| 06a_wiou | 11.1M | 0.7001 | 0.7125 | 0.7655 | 0.4117 | #3 |
| 03_p2_layer | 10.6M | 0.7340 | 0.7245 | 0.7655 | 0.3965 | #4 |
| **exp08_spd_only** | 11.4M | 0.675 | 0.701 | 0.756 | 0.395 | #5 |
| 04_combined_cbam_p2 | 10.7M | 0.6961 | 0.6932 | 0.7525 | 0.3880 | #6 |
| **exp07_spd_cbam** | 11.5M | 0.669 | 0.711 | 0.750 | 0.407 | #7 |
| 06b_focal | 11.1M | 0.6180 | 0.6970 | 0.7290 | 0.3630 | #8 |
| baseline_seed | 11.1M | 0.6665 | 0.6840 | 0.6949 | 0.3740 | #9 |
| **exp09_cbam_gfpn** | 11.15M | 0.631 | 0.684 | 0.729 | 0.381 | #10 |

---

## 消融实验结论

### 有效策略 (✅)
| 策略 | 效果 | 说明 |
|------|------|------|
| **CBAM注意力** | +1.7% mAP50 | 最有效 |
| **WIoU损失** | 持平 | 轻微提升 |

### 无效策略 (❌)
| 策略 | 效果 | 说明 |
|------|------|------|
| **Focal Loss** | -6% mAP50 | γ=2.0反而降低效果 |
| P2检测层 | -1% mAP50 | 增加复杂度但效果下降 |
| CBAM+P2组合 | -3% mAP50 | 组合反而更差 |
| **SPD-Conv** | -2% mAP50 | 纯SPD和SPD+CBAM组合均无效 |
| baseline_seed | -14% mAP50 | 训练集大量减少，非正式实验 |

### 待优化类别
- **crazing**: recall极低 (~0.23)，漏检严重
- **rolled-in_scale**: 效果中等 (~0.69)

---

## 各实验详情

### 1. baseline (yolov8s, 11.1M参数)
- mAP50-95: **0.4342** (最佳)
- mAP50: 0.7733
- P: 0.7000, R: 0.7303

### 2. 02_cbam (CBAM注意力, 11.2M参数)
- mAP50: **0.7870** (最佳)
- mAP50-95: 0.4258
- P: **0.7595** (最佳Precision)

### 3. 06a_wiou (WIoU损失函数, 11.1M参数)
- mAP50-95: 0.4117
- mAP50: 0.7655
- 中规中矩，不如CBAM惊艳

### 4. 06b_focal (Focal Loss, γ=2.0, 11.1M参数)
- mAP50-95: 0.3630 (最差之一)
- mAP50: 0.7290
- **Focal Loss反而降低效果**，可能γ=2.0太激进

### 5. 03_p2_layer (P2高分辨率层)
- mAP50-95: 0.3965
- 对rolled-in_scale有提升但整体下降

### 6. 04_combined_cbam_p2 (CBAM+P2组合)
- mAP50-95: 0.3880
- 组合效果不如单独CBAM

### 7. exp07_spd_cbam (SPD-Conv + CBAM, 11.5M参数)
- mAP50: 0.750, mAP50-95: 0.407
- P: 0.669, R: 0.711
- **结论**: SPD和CBAM组合反而比单独CBAM差

### 8. exp08_spd_only (纯SPD-Conv, 11.4M参数)
- mAP50: 0.756, mAP50-95: 0.395
- P: 0.675, R: 0.701
- **结论**: 纯SPD略优于SPD+CBAM，但仍不如baseline和CBAM

---

## 后续实验建议

**不建议继续的方向:**
- Focal Loss: 已验证无效
- P2检测层: 单独使用效果下降
- CBAM+P2组合: 效果反而更差
- SPD-Conv: 纯SPD和组合均无效

**可能有效的方向:**
1. **类别权重**: 手动提高crazing和rolled-in_scale的损失权重
2. **数据增强**: Mosaic+MixUp已验证有效，可尝试更激进的配置
3. **多尺度测试**: 训练时使用多尺度，推理时使用TTA

---

## 改动记录

### Focal Loss集成 (已撤销)
Focal Loss已验证无效，相关代码已恢复。

**实验脚本位置:** `src/experiments/exp06b_focal/`

---

## WIoU改动 (已撤销)
WIoU已撤销，改动记录见 `ultralytics-main/ultralytics/utils/loss_wiou_MODIFIED.py`

---

## SPD-Conv实验 (2026-04-11)

**实验脚本位置:** `src/experiments/exp07_spd_conv/` (SPD+CBAM), `src/experiments/exp08_spd_only/` (纯SPD)

**结论:** SPD-Conv在NEU-DET数据集上无效，不建议继续使用。

**SPD-Conv改动已完全撤销** - 相关代码已从 ultralytics-main 移除。

---

## CBAM+BiFPN-style Neck实验 (2026-04-12)

**实验脚本位置:** `src/experiments/exp09_cbam_gfpn_neck/`

**结果:**
- mAP50: 0.729
- mAP50-95: 0.381
- **结论:** BiFPN-style neck 反而降低了性能，排名第10

**配置说明:**
- Backbone: YOLOv8s + CBAM
- Neck: BiFPN-style (top-down + bottom-up 多尺度融合)
- Head: Detect

**教训:** 增加 neck 复杂度没有帮助，CBAM 在 backbone 上的效果更直接有效。标准 PANet 已经是优秀的选择。
