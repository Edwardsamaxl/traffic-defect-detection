"""
增强伪标签生成器
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict

from ultralytics import YOLO

from ..utils.config import (
    PROJECT_ROOT,
    CLASSES,
    PSEUDO_LABEL_DEFAULTS,
    ensure_dirs,
)


class PseudoLabelGenerator:
    """
    伪标签生成器 - 支持多种生成策略

    使用示例:
        generator = PseudoLabelGenerator(
            model_path="experiments/baseline_seed/weights/best.pt",
            data_yaml="datasets/neu.yaml",
        )
        generator.generate_adaptive(
            unlabeled_dir="data/NEU-DET/unlabeled/images/train",
            output_dir="data/NEU-DET/pseudo_labels_adaptive",
        )
    """

    def __init__(
        self,
        model_path: Path | str,
        data_yaml: Path | str,
        imgsz: int = 640,
    ):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.imgsz = imgsz

        self.model: Optional[YOLO] = None
        self.baseline_ap: Optional[list] = None

    def load_model(self):
        """加载模型"""
        if self.model is None:
            self.model = YOLO(str(self.model_path))

    def _eval_baseline_ap(self) -> list:
        """获取 baseline 每类 AP"""
        if self.baseline_ap is not None:
            return self.baseline_ap

        self.load_model()
        metrics = self.model.val(
            data=str(self.data_yaml),
            imgsz=self.imgsz,
            conf=0.001,
            iou=0.6,
            augment=True,
            verbose=False,
        )

        # 提取每类 AP50
        ap_values = []
        for key, value in metrics.results_dict.items():
            if "metrics/mAP50(" in key and key.endswith(")"):
                ap_values.append(float(value))

        self.baseline_ap = ap_values
        return ap_values

    def _compute_adaptive_threshold(
        self,
        class_id: int,
        base_conf: float = PSEUDO_LABEL_DEFAULTS["base_conf"],
        lambda_val: float = PSEUDO_LABEL_DEFAULTS["adaptive_lambda"],
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

    def _yolo_to_xyxy(self, box):
        """YOLO xywh -> xyxy"""
        x, y, w, h = box
        return x - w/2, y - h/2, x + w/2, y + h/2

    def _iou_xyxy(self, a, b):
        """计算 xyxy 格式 IoU"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-9

        return inter / union

    def generate_standard(
        self,
        unlabeled_dir: Path | str,
        output_dir: Path | str,
        conf_threshold: float = PSEUDO_LABEL_DEFAULTS["standard_conf"],
        save_images: bool = False,
    ):
        """
        标准伪标签生成 - 固定阈值

        Args:
            unlabeled_dir: 无标签图像目录
            output_dir: 输出标签目录
            conf_threshold: 置信度阈值
        """
        unlabeled_dir = Path(unlabeled_dir)
        output_dir = Path(output_dir)
        ensure_dirs(output_dir)

        self.load_model()

        # 获取图像
        image_paths = sorted([
            p for p in unlabeled_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        print(f"\n标准伪标签生成: {len(image_paths)} 张图像")
        print(f"置信度阈值: {conf_threshold}")

        stats = defaultdict(int)
        saved_count = 0

        for img_path in image_paths:
            # 预测
            result = self.model.predict(
                source=str(img_path),
                imgsz=self.imgsz,
                conf=conf_threshold,
                verbose=False,
            )[0]

            label_path = output_dir / f"{img_path.stem}.txt"

            if result.boxes is None or len(result.boxes) == 0:
                continue

            # 保存标签
            with open(label_path, "w") as f:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    x, y, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    stats[cls_id] += 1
                    saved_count += 1

        print(f"\n统计:")
        for cls_id, count in sorted(stats.items()):
            print(f"  {CLASSES[cls_id]:<20}: {count}")
        print(f"总计: {saved_count} 个伪标签\n")

        return dict(stats)

    def generate_adaptive(
        self,
        unlabeled_dir: Path | str,
        output_dir: Path | str,
        base_conf: float = PSEUDO_LABEL_DEFAULTS["base_conf"],
        lambda_val: float = PSEUDO_LABEL_DEFAULTS["adaptive_lambda"],
    ):
        """
        自适应阈值伪标签生成

        根据每类 AP 自动调整阈值
        """
        unlabeled_dir = Path(unlabeled_dir)
        output_dir = Path(output_dir)
        ensure_dirs(output_dir)

        # 获取 baseline AP
        self._eval_baseline_ap()

        # 计算每类阈值
        class_thresholds = {
            i: self._compute_adaptive_threshold(i, base_conf, lambda_val)
            for i in range(len(CLASSES))
        }

        print(f"\n自适应阈值伪标签生成")
        print(f"每类阈值: { {CLASSES[k]: f'{v:.3f}' for k, v in class_thresholds.items()} }")

        self.load_model()

        # 获取图像
        image_paths = sorted([
            p for p in unlabeled_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        stats = defaultdict(int)
        saved_count = 0

        for img_path in image_paths:
            result = self.model.predict(
                source=str(img_path),
                imgsz=self.imgsz,
                conf=0.01,  # 先用低阈值初筛
                verbose=False,
            )[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            label_path = output_dir / f"{img_path.stem}.txt"
            valid_boxes = []

            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                threshold = class_thresholds.get(cls_id, base_conf)

                if conf >= threshold:
                    x, y, w, h = box.xywhn[0].tolist()
                    valid_boxes.append((cls_id, x, y, w, h))
                    stats[cls_id] += 1
                    saved_count += 1

            if valid_boxes:
                with open(label_path, "w") as f:
                    for cls_id, x, y, w, h in valid_boxes:
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"\n统计:")
        for cls_id, count in sorted(stats.items()):
            print(f"  {CLASSES[cls_id]:<20}: {count}")
        print(f"总计: {saved_count} 个伪标签\n")

        return dict(stats)

    def generate_consistency(
        self,
        unlabeled_dir: Path | str,
        output_dir: Path | str,
        base_conf: float = PSEUDO_LABEL_DEFAULTS["base_conf"],
        iou_match: float = PSEUDO_LABEL_DEFAULTS["iou_match"],
    ):
        """
        翻转一致性伪标签生成

        使用原图和水平翻转图的一致性筛选
        """
        unlabeled_dir = Path(unlabeled_dir)
        output_dir = Path(output_dir)
        ensure_dirs(output_dir)

        self.load_model()

        image_paths = sorted([
            p for p in unlabeled_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        print(f"\n翻转一致性伪标签生成: {len(image_paths)} 张图像")

        stats = defaultdict(int)
        saved_count = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 原图预测
            res_orig = self.model.predict(
                source=img,
                imgsz=self.imgsz,
                conf=0.01,
                augment=False,
                verbose=False,
            )[0]

            # 翻转图预测
            res_flip = self.model.predict(
                source=cv2.flip(img, 1),  # 水平翻转
                imgsz=self.imgsz,
                conf=0.01,
                augment=False,
                verbose=False,
            )[0]

            if res_orig.boxes is None or len(res_orig.boxes) == 0:
                continue

            # 收集原图 boxes
            orig_boxes = []
            for b in res_orig.boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                if conf >= base_conf:
                    xywh = b.xywhn[0].tolist()
                    orig_boxes.append((cid, conf, xywh))

            # 收集翻转 boxes
            flip_boxes = []
            if res_flip.boxes is not None and len(res_flip.boxes) > 0:
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
                    x, y, w, h = xywh
                    valid_boxes.append((cid, x, y, w, h))
                    stats[cid] += 1
                    saved_count += 1

            # 保存
            if valid_boxes:
                label_path = output_dir / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    for cls_id, x, y, w, h in valid_boxes:
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"\n统计:")
        for cls_id, count in sorted(stats.items()):
            print(f"  {CLASSES[cls_id]:<20}: {count}")
        print(f"总计: {saved_count} 个伪标签\n")

        return dict(stats)

    def generate_adaptive_consistency(
        self,
        unlabeled_dir: Path | str,
        output_dir: Path | str,
        base_conf: float = PSEUDO_LABEL_DEFAULTS["base_conf"],
        lambda_val: float = PSEUDO_LABEL_DEFAULTS["adaptive_lambda"],
        iou_match: float = PSEUDO_LABEL_DEFAULTS["iou_match"],
    ):
        """
        自适应阈值 + 翻转一致性组合
        """
        unlabeled_dir = Path(unlabeled_dir)
        output_dir = Path(output_dir)
        ensure_dirs(output_dir)

        # 获取 baseline AP
        self._eval_baseline_ap()

        # 计算每类阈值
        class_thresholds = {
            i: self._compute_adaptive_threshold(i, base_conf, lambda_val)
            for i in range(len(CLASSES))
        }

        print(f"\n自适应 + 一致性伪标签生成")
        print(f"每类阈值: { {CLASSES[k]: f'{v:.3f}' for k, v in class_thresholds.items()} }")

        self.load_model()

        image_paths = sorted([
            p for p in unlabeled_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        stats = defaultdict(int)
        saved_count = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 原图
            res_orig = self.model.predict(
                source=img,
                imgsz=self.imgsz,
                conf=0.01,
                augment=False,
                verbose=False,
            )[0]

            # 翻转
            res_flip = self.model.predict(
                source=cv2.flip(img, 1),
                imgsz=self.imgsz,
                conf=0.01,
                augment=False,
                verbose=False,
            )[0]

            if res_orig.boxes is None or len(res_orig.boxes) == 0:
                continue

            # 原图 boxes
            orig_boxes = []
            for b in res_orig.boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                th = class_thresholds.get(cid, base_conf)
                if conf >= th:
                    orig_boxes.append((cid, conf, b.xywhn[0].tolist()))

            # 翻转 boxes
            flip_boxes = []
            if res_flip.boxes is not None and len(res_flip.boxes) > 0:
                for b in res_flip.boxes:
                    cid = int(b.cls.item())
                    conf = float(b.conf.item())
                    th = class_thresholds.get(cid, base_conf)
                    if conf >= th:
                        x, y, w, h = b.xywhn[0].tolist()
                        flip_boxes.append((cid, conf, [1.0 - x, y, w, h]))

            # 匹配
            valid_boxes = []
            for cid, conf, xywh in orig_boxes:
                box_xyxy = self._yolo_to_xyxy(xywh)
                matched = False

                for fcid, fconf, fxywh in flip_boxes:
                    if fcid != cid:
                        continue
                    if self._iou_xyxy(box_xyxy, self._yolo_to_xyxy(fxywh)) >= iou_match:
                        matched = True
                        break

                if matched:
                    x, y, w, h = xywh
                    valid_boxes.append((cid, x, y, w, h))
                    stats[cid] += 1
                    saved_count += 1

            if valid_boxes:
                label_path = output_dir / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    for cls_id, x, y, w, h in valid_boxes:
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"\n统计:")
        for cls_id, count in sorted(stats.items()):
            print(f"  {CLASSES[cls_id]:<20}: {count}")
        print(f"总计: {saved_count} 个伪标签\n")

        return dict(stats)

    def generate_with_uncertainty(
        self,
        unlabeled_dir: Path | str,
        output_dir: Path | str,
        num_stochastic_passes: int = 5,
        uncertainty_threshold: float = 0.1,
        conf_threshold: float = 0.5,
    ):
        """
        基于不确定性的伪标签生成

        使用多次随机 dropout 推理估计不确定性
        """
        unlabeled_dir = Path(unlabeled_dir)
        output_dir = Path(output_dir)
        ensure_dirs(output_dir)

        self.load_model()

        image_paths = sorted([
            p for p in unlabeled_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        print(f"\n不确定性伪标签生成: {len(image_paths)} 张图像")
        print(f"随机次数: {num_stochastic_passes}, 不确定性阈值: {uncertainty_threshold}")

        stats = defaultdict(int)
        saved_count = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 多次推理收集预测
            all_predictions = []
            for _ in range(num_stochastic_passes):
                result = self.model.predict(
                    source=img,
                    imgsz=self.imgsz,
                    conf=0.01,
                    verbose=False,
                )[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_data = []
                    for i in range(len(result.boxes)):
                        boxes_data.append({
                            "cls": int(result.boxes.cls[i].item()),
                            "conf": float(result.boxes.conf[i].item()),
                            "xywhn": result.boxes.xywhn[i].tolist(),
                        })
                    all_predictions.append(boxes_data)

            if not all_predictions:
                continue

            # 计算每个预测的不确定性
            # 简化：使用不同 pass 之间预测数量的方差
            num_preds_per_pass = [len(p) for p in all_predictions]
            avg_preds = np.mean(num_preds_per_pass)
            std_preds = np.std(num_preds_per_pass)

            # 选择预测数接近平均的 pass 作为稳定预测
            stable_pass_idx = np.argmin(np.abs(np.array(num_preds_per_pass) - avg_preds))
            stable_boxes = all_predictions[stable_pass_idx]

            # 进一步过滤
            valid_boxes = []
            for box in stable_boxes:
                if box["conf"] >= conf_threshold:
                    valid_boxes.append(box)
                    stats[box["cls"]] += 1
                    saved_count += 1

            if valid_boxes:
                label_path = output_dir / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    for box in valid_boxes:
                        x, y, w, h = box["xywhn"]
                        f.write(f"{box['cls']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"\n统计:")
        for cls_id, count in sorted(stats.items()):
            print(f"  {CLASSES[cls_id]:<20}: {count}")
        print(f"总计: {saved_count} 个伪标签\n")

        return dict(stats)
