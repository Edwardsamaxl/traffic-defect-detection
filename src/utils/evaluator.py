"""
增强评估工具 - 包含混淆矩阵、PR曲线、错误分析
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import (
    PROJECT_ROOT,
    REPORTS_ROOT,
    CLASSES,
    ensure_dirs,
)


class EnhancedEvaluator:
    """
    增强评估器 - 支持混淆矩阵、PR曲线、错误分析

    使用示例:
        evaluator = EnhancedEvaluator("baseline", "experiments/baseline/weights/best.pt")
        evaluator.evaluate()
        evaluator.plot_confusion_matrix()
        evaluator.plot_pr_curves()
        evaluator.save_failure_cases()
    """

    def __init__(
        self,
        experiment_name: str,
        model_path: Path | str,
        data_yaml: Path | str,
        split: str = "test",
        conf: float = 0.001,
        iou: float = 0.6,
        imgsz: int = 640,
    ):
        self.experiment_name = experiment_name
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.split = split
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        # 输出目录
        self.output_dir = REPORTS_ROOT / "evaluations" / experiment_name
        ensure_dirs(self.output_dir)

        # 模型
        self.model: Optional[YOLO] = None
        self.metrics: Optional[Any] = None

        # 存储预测结果用于分析
        self.all_predictions: list[dict] = []
        self.all_ground_truths: list[dict] = []

        # 混淆矩阵
        self.confusion_matrix: Optional[np.ndarray] = None

    def load_model(self):
        """加载模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def evaluate(self, tta: bool = False) -> dict:
        """
        执行评估

        Returns:
            评估结果字典
        """
        self.load_model()

        print(f"\n{'='*60}")
        print(f"评估: {self.experiment_name}")
        print(f"模型: {self.model_path}")
        print(f"{'='*60}\n")

        # 执行验证
        self.metrics = self.model.val(
            data=str(self.data_yaml),
            split=self.split,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            augment=tta,
            verbose=True,
            save_json=True,  # 保存 COCO 格式结果
        )

        results = self.metrics.results_dict

        # 提取每类 AP
        per_class_ap = {}
        for key, value in results.items():
            if "metrics/mAP50(" in key:
                cls_name = key.split("(")[1].rstrip(")")
                per_class_ap[cls_name] = float(value)

        self.per_class_ap = per_class_ap

        # 保存结果
        self._save_results(results)

        return results

    def _save_results(self, results: dict):
        """保存评估结果到 JSON"""
        output_file = self.output_dir / "evaluation_results.json"

        # 格式化结果
        output = {
            "experiment": self.experiment_name,
            "model": str(self.model_path),
            "config": {
                "split": self.split,
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": self.imgsz,
            },
            "metrics": {
                "precision": float(results.get("metrics/precision(B)", 0)),
                "recall": float(results.get("metrics/recall(B)", 0)),
                "map50": float(results.get("metrics/mAP50(B)", 0)),
                "map50_95": float(results.get("metrics/mAP50-95(B)", 0)),
            },
            "per_class_ap": self.per_class_ap,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"结果已保存: {output_file}")

    def analyze_predictions(self, save_images: bool = True):
        """
        分析预测结果 - 收集所有预测和真值用于后续分析
        """
        if self.model is None:
            self.load_model()

        # 获取验证集图像路径
        import yaml
        with open(self.data_yaml) as f:
            data = yaml.safe_load(f)

        val_dir = PROJECT_ROOT / data[self.split].replace("../", "")
        image_files = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))

        print(f"\n分析 {len(image_files)} 张图像...")

        predictions = []
        ground_truths = []

        for img_path in image_files:
            # 读取真值标签
            label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt"))
            gt_boxes = []
            if label_path.exists():
                with open(label_path) as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:])
                            gt_boxes.append({"class_id": cls_id, "bbox": [x, y, w, h]})

            # 预测
            result = self.model.predict(source=str(img_path), conf=self.conf, verbose=False)[0]
            pred_boxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    pred_boxes.append({
                        "class_id": int(result.boxes.cls[i].item()),
                        "confidence": float(result.boxes.conf[i].item()),
                        "bbox": result.boxes.xywhn[i].tolist(),  # 归一化坐标
                    })

            predictions.append({"image": img_path.name, "boxes": pred_boxes})
            ground_truths.append({"image": img_path.name, "boxes": gt_boxes})

            # 保存预测图像
            if save_images:
                self._save_annotated(img_path, result, img_path.stem)

        self.all_predictions = predictions
        self.all_ground_truths = ground_truths

        # 保存分析结果
        analysis_file = self.output_dir / "prediction_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump({
                "predictions": predictions,
                "ground_truths": ground_truths,
            }, f, indent=2, ensure_ascii=False)

        print(f"预测分析已保存: {analysis_file}")

    def _save_annotated(self, img_path: Path, result, stem: str):
        """保存带标注的图像"""
        plotted = result.plot()
        output_path = self.output_dir / "annotated" / f"{stem}.jpg"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), plotted)

    def compute_confusion_matrix(
        self,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.001,
    ):
        """
        计算混淆矩阵

        Args:
            iou_threshold: IoU 匹配阈值
            conf_threshold: 置信度阈值
        """
        if not self.all_predictions:
            self.analyze_predictions(save_images=False)

        n_classes = len(CLASSES)
        # 混淆矩阵: 行=真值, 列=预测
        cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int32)  # +1 for background

        for pred, gt in zip(self.all_predictions, self.all_ground_truths):
            # 匹配预测和真值
            matched_gt = set()
            matched_pred = set()

            # 按置信度排序预测
            sorted_preds = sorted(
                enumerate(pred["boxes"]),
                key=lambda x: x[1]["confidence"],
                reverse=True
            )

            for pred_idx, pred_box in sorted_preds:
                if pred_box["confidence"] < conf_threshold:
                    continue

                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(gt["boxes"]):
                    if gt_idx in matched_gt:
                        continue
                    if gt_box["class_id"] != pred_box["class_id"]:
                        continue

                    iou = self._compute_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    # 真正例
                    cm[pred_box["class_id"], gt_box["class_id"]] += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                else:
                    # 假正例
                    cm[pred_box["class_id"], n_classes] += 1

            # 假负例（未匹配的gt）
            for gt_idx, gt_box in enumerate(gt["boxes"]):
                if gt_idx not in matched_gt:
                    cm[n_classes, gt_box["class_id"]] += 1

        self.confusion_matrix = cm
        return cm

    def _compute_iou(self, box1, box2):
        """计算两个归一化 bbox 的 IoU (xywh format)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 转 xyxy
        x1_min = x1 - w1/2
        y1_min = y1 - h1/2
        x1_max = x1 + w1/2
        y1_max = y1 + h1/2

        x2_min = x2 - w2/2
        y2_min = y2 - h2/2
        x2_max = x2 + w2/2
        y2_max = y2 + h2/2

        # 相交区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        # 各自面积
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-9)

    def plot_confusion_matrix(self, save: bool = True):
        """绘制混淆矩阵"""
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()

        cm = self.confusion_matrix

        # 计算百分比
        cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        # 保存 numpy 数组
        np.save(self.output_dir / "confusion_matrix.npy", cm)

        # 保存 CSV
        cm_file = self.output_dir / "confusion_matrix.csv"
        with open(cm_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 表头
            header = [""] + CLASSES + ["FN"]
            writer.writerow(header)
            # 数据行
            for i in range(n_classes := len(CLASSES)):
                row = [CLASSES[i]]
                for j in range(n_classes):
                    row.append(int(cm[i, j]))
                row.append(int(cm[i, n_classes]))
                writer.writerow(row)
            # FN 总计
            fn_row = ["FN"] + [int(cm[n_classes, j]) for j in range(n_classes)] + [""]
            writer.writerow(fn_row)

        print(f"混淆矩阵已保存: {cm_file}")
        return cm

    def save_failure_cases(
        self,
        output_subdir: str = "failure_cases",
        max_cases: int = 50,
    ):
        """
        保存失败案例（漏检和误检）

        Args:
            output_subdir: 输出子目录
            max_cases: 最大保存案例数
        """
        if not self.all_predictions:
            self.analyze_predictions(save_images=True)

        failure_dir = self.output_dir / output_subdir
        ensure_dirs(failure_dir)

        fn_dir = failure_dir / "false_negatives"  # 漏检
        fp_dir = failure_dir / "false_positives"   # 误检
        fn_dir.mkdir(exist_ok=True)
        fp_dir.mkdir(exist_ok=True)

        fn_count = 0
        fp_count = 0

        for pred, gt in zip(self.all_predictions, self.all_ground_truths):
            if fn_count >= max_cases and fp_count >= max_cases:
                break

            img_name = pred["image"]

            # 找出漏检（gt有但pred没有匹配）
            for gt_box in gt["boxes"]:
                matched = False
                for pred_box in pred["boxes"]:
                    iou = self._compute_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou >= 0.5 and pred_box["class_id"] == gt_box["class_id"]:
                        matched = True
                        break
                if not matched and fn_count < max_cases:
                    # 保存漏检信息
                    info = {
                        "image": img_name,
                        "true_class": CLASSES[gt_box["class_id"]],
                        "bbox": gt_box["bbox"],
                    }
                    with open(fn_dir / f"{img_name.stem}_info.json", "w") as f:
                        json.dump(info, f, indent=2)
                    fn_count += 1

            # 找出误检（pred有但gt没有匹配）
            for pred_box in pred["boxes"]:
                if pred_box["confidence"] < 0.1:  # 只看高置信度误检
                    continue
                matched = False
                for gt_box in gt["boxes"]:
                    if gt_box["class_id"] != pred_box["class_id"]:
                        continue
                    iou = self._compute_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou >= 0.5:
                        matched = True
                        break
                if not matched and fp_count < max_cases:
                    info = {
                        "image": img_name,
                        "predicted_class": CLASSES[pred_box["class_id"]],
                        "confidence": pred_box["confidence"],
                        "bbox": pred_box["bbox"],
                    }
                    with open(fp_dir / f"{img_name.stem}_info.json", "w") as f:
                        json.dump(info, f, indent=2)
                    fp_count += 1

        print(f"失败案例已保存:")
        print(f"  漏检 (FN): {fn_dir} ({fn_count} cases)")
        print(f"  误检 (FP): {fp_dir} ({fp_count} cases)")

    def print_class_summary(self):
        """打印类别级别总结"""
        print(f"\n{'='*60}")
        print(f"类别级别 mAP@0.5")
        print(f"{'='*60}")
        for cls_name, ap in sorted(self.per_class_ap.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(ap * 20)
            print(f"  {cls_name:<20} {ap:.4f} {bar}")
        print(f"{'='*60}\n")


def batch_evaluate(
    experiments: list[dict],
    output_csv: Optional[Path] = None,
):
    """
    批量评估多个实验

    Args:
        experiments: 实验列表，每个元素包含 name, model_path, data_yaml
        output_csv: 输出 CSV 路径
    """
    rows = []

    for exp in experiments:
        evaluator = EnhancedEvaluator(
            experiment_name=exp["name"],
            model_path=exp["model_path"],
            data_yaml=exp["data_yaml"],
        )

        try:
            results = evaluator.evaluate()
            evaluator.print_class_summary()

            rows.append({
                "experiment": exp["name"],
                "precision": results.get("metrics/precision(B)", 0),
                "recall": results.get("metrics/recall(B)", 0),
                "map50": results.get("metrics/mAP50(B)", 0),
                "map50_95": results.get("metrics/mAP50-95(B)", 0),
            })
        except Exception as e:
            print(f"[ERROR] 评估 {exp['name']} 失败: {e}")
            rows.append({
                "experiment": exp["name"],
                "error": str(e),
            })

    # 保存 CSV
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"\n批量评估结果已保存: {output_csv}")

    return rows
