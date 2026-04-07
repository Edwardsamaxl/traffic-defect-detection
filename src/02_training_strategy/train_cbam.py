"""
CBAM注意力机制训练策略
=====================================
优化目标: 增强模型对缺陷特征的提取能力

参考论文:
- CBAM: Convolutional Block Attention Module (ECCV 2018)
  https://arxiv.org/abs/1807.06521
  Woo, S., et al.

- Steel Surface Defect Detection Based on Improved YOLOv8 with Multi-Scale Feature Fusion
  and Attention Mechanism (MDPI 2026)
  https://www.mdpi.com/2079-9292/15/7/1408

- SLF-YOLO: Enhanced YOLOv8 Model for Metal Surface Defect Detection (Scientific Reports 2025)
  https://www.nature.com/articles/s41598-025-94936-9

理论依据:
CBAM包含两个顺序的子模块:
1. 通道注意力模块(CAM): 通过MaxPool和AvgPool提取通道统计信息，经MLP生成通道权重
2. 空间注意力模块(SAM): 通过通道压缩后的空间关联性生成空间权重

对于钢铁表面缺陷:
- 通道注意力可以帮助区分不同缺陷类型
- 空间注意力可以定位缺陷区域

配置文件:
- 模型配置: ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam.yaml
- CBAM模块: ultralytics-main/ultralytics/nn/modules/conv.py

前置要求:
- tasks.py 需要添加 CBAM 模块导入
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_CFG = ROOT / "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam.yaml"
DATA_YAML = ROOT / "datasets/neu.yaml"


def main():
    """训练CBAM增强模型"""
    model = YOLO(str(MODEL_CFG))

    model.train(
        data=str(DATA_YAML),
        epochs=200,
        patience=50,
        imgsz=640,
        batch=4,
        project=str(ROOT / "experiments"),
        name="02_cbam",
        # 数据增强配置 - 与baseline一致
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        # 损失权重 - 与baseline一致
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )


if __name__ == "__main__":
    main()
