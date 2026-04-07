"""
WIoU Loss Implementation for YOLOv8
=====================================

参考论文:
- Focal and Efficient IoU Loss: Accelerating Learning and Improving Object Detection Performance
  https://arxiv.org/abs/2211.06305
  Tong, Z., et al.

- Improved YOLOv8 Model for Strip Steel Surface Defect Detection (MDPI 2024)
  https://www.mdpi.com/2076-3417/15/1/52

- MPA-YOLO: Steel Surface Defect Detection Based on Improved YOLOv8 Framework (Pattern Recognition 2025)
  https://www.sciencedirect.com/science/article/pii/S0031320325005576

理论依据:
WIoU (Wise-IoU) Loss 由 Tong 等人提出，通过引入注意力机制来聚焦于"困难样本"。
相比 CIoU Loss，WIoU 可以更好地处理不同尺度和长宽比的边界框。

WIoU v1: 基础版本，引入IoU注意力
WIoU v2: 加入时序衰减机制
WIoU v3: 梯度无痛更新，保持训练稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou


class WIoUYv3Loss(nn.Module):
    """Wise-IoU Loss v3 - 梯度无痛版本

    该版本通过构建单调聚焦系数来减弱"困难样本"对训练的干扰，
    同时保留"困难样本"的有用梯度信息。

    属性:
        beta (float): 聚焦系数，用于控制对离群样本的惩罚程度
        label_smooth (float): 标签平滑系数
    """

    def __init__(self, beta: float = 2.0, label_smooth: float = 0.0):
        """初始化 WIoU Loss v3

        Args:
            beta: 聚焦系数，默认为2.0。值越小，对离群样本的惩罚越轻
            label_smooth: 标签平滑系数，默认为0.0
        """
        super().__init__()
        self.beta = beta
        self.label_smooth = label_smooth

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_scores: torch.Tensor = None,
        fg_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """计算 WIoU Loss

        Args:
            pred_boxes: 预测边界框 [N, 4] (xywh format)
            target_boxes: 目标边界框 [N, 4] (xywh format)
            target_scores: 目标分数 [N, nc] (可选)
            fg_mask: 正样本掩码 [N] (可选)

        Returns:
            torch.Tensor: WIoU损失值
        """
        if fg_mask is not None:
            pred_boxes = pred_boxes[fg_mask]
            target_boxes = target_boxes[fg_mask]

        # 计算IoU
        iou = bbox_iou(pred_boxes, target_boxes, xywh=True, CIoU=False)
        if self.label_smooth > 0:
            iou = iou * (1 - self.label_smooth) + self.label_smooth / (iou.numel() + 1e-7)

        # 计算IoU注意力
        with torch.no_grad():
            # 构建单调聚焦系数
           beta = self.beta
            # 防止beta为0或负数
            beta = max(beta, 1e-6)

            # 计算聚焦权重
            if beta == 1.0:
                weight = torch.ones_like(iou)
            else:
                # WIoU v3 的核心公式
                weight = (iou.pow(beta) + 1e-7) / ((1 - iou).pow(beta) + 1e-7)

            # 归一化权重
            weight_sum = weight.sum()
            if weight_sum > 0:
                weight = weight / weight_sum

        # 计算WIoU Loss
        loss = (1 - iou) * weight

        return loss.sum()


class WIoUv3BboxLoss(nn.Module):
    """WIoU v3 边界框损失模块

    结合 WIoU Loss 和 DFL (Distribution Focal Loss) 进行边界框回归
    """

    def __init__(self, reg_max: int = 16, beta: float = 2.0):
        """初始化 WIoU Bbox Loss

        Args:
            reg_max: DFL 最大通道数
            beta: WIoU 聚焦系数
        """
        super().__init__()
        self.reg_max = reg_max
        self.wiou_loss = WIoUYv3Loss(beta=beta)
        self.dfl_loss = None if reg_max <= 1 else self._init_dfl()

    def _init_dfl(self):
        """初始化 DFL Loss"""
        from ultralytics.utils.loss import DFLoss
        return DFLoss(self.reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 WIoU 和 DFL 损失

        Args:
            pred_dist: 预测分布 [B, N, 4*reg_max]
            pred_bboxes: 预测边界框 [B, N, 4]
            anchor_points: 锚点 [B, N, 2]
            target_bboxes: 目标边界框 [B, N, 4]
            target_scores: 目标分数 [B, N, nc]
            target_scores_sum: 目标分数总和 [B]
            fg_mask: 正样本掩码 [B, N]
            imgsz: 图像尺寸 [2]
            stride: 步长 [B]

        Returns:
            tuple: (loss_iou, loss_dfl)
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # WIoU Loss
        loss_iou = self.wiou_loss(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            fg_mask=fg_mask[fg_mask] if fg_mask.any() else None
        )

        # 如果有DDF Loss，计算DFL Loss
        if self.dfl_loss is not None:
            from ultralytics.utils.tal import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


def create_wiou_loss_module(beta: float = 2.0, reg_max: int = 16):
    """创建 WIoU Loss 模块的工厂函数

    Args:
        beta: 聚焦系数
        reg_max: DFL 最大通道数

    Returns:
        WIoUv3BboxLoss: WIoU边界框损失模块
    """
    return WIoUv3BboxLoss(reg_max=reg_max, beta=beta)
