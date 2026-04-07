# WIoU Loss Modifications - Original Code Backup
# =============================================
# This file contains the ORIGINAL code before WIoU modifications.
# Restore these if needed.

# File: ultralytics/utils/loss.py
# Class: BboxLoss (lines ~109-154)

"""
ORIGINAL BboxLoss.__init__:
```python
def __init__(self, reg_max: int = 16):
    super().__init__()
    self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
```

ORIGINAL BboxLoss.forward (IoU part):
```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

# Class: v8DetectionLoss (lines ~365-391)

ORIGINAL v8DetectionLoss.__init__:
```python
def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
    # ...
    self.bbox_loss = BboxLoss(m.reg_max).to(device)
```

# File: ultralytics/nn/tasks.py
# Method: DetectionModel.init_criterion (lines ~515-517)

ORIGINAL init_criterion:
```python
def init_criterion(self):
    return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
```

# =============================================
# MODIFIED CODE (Current)
# =============================================

# File: ultralytics/utils/loss.py

class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16, use_wiou: bool = False, beta: float = 2.0):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.use_wiou = use_wiou
        self.beta = beta

    def forward(self, ...):
        if self.use_wiou:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=True, CIoU=False)
            with torch.no_grad():
                beta = self.beta
                beta = max(beta, 1e-6)
                weight_iou = (iou.pow(beta) + 1e-7) / ((1 - iou).pow(beta) + 1e-7)
                weight_sum = weight_iou.sum()
                if weight_sum > 0:
                    weight_iou = weight_iou / weight_sum
            loss_iou = ((1.0 - iou) * weight_iou * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        # ... rest unchanged

class v8DetectionLoss:
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, use_wiou: bool = False):
        # ...
        self.use_wiou = use_wiou
        self.bbox_loss = BboxLoss(m.reg_max, use_wiou=use_wiou).to(device)

# File: ultralytics/nn/tasks.py

def init_criterion(self):
    use_wiou = getattr(self, "use_wiou", False) or "wiou" in (getattr(self, "name", "") or "").lower()
    return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self, use_wiou=use_wiou)
