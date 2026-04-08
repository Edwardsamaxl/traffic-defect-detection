"""
伪标签生成脚本 - 支持三种策略

策略选择 (通过 --strategy 参数):
    1. standard     - 固定置信度阈值 (CONF_THRES=0.7)
    2. adaptive     - 自适应阈值 (根据每类AP调整)
    3. adaptive_consistency - 自适应阈值 + 翻转一致性

用法:
    python src/04_semi_supervised/generate_pseudo_labels.py --strategy adaptive_consistency
"""
import argparse
import cv2
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "experiments/baseline_seed/weights/baseline_seed.pt"
DATA_YAML = ROOT / "datasets/neu.yaml"
UNLABELED_IMG_DIR = ROOT / "data/NEU-DET/unlabeled/images/train"
PSEUDO_LABEL_DIR = ROOT / "data/NEU-DET/unlabeled/pseudo_labels"

IMG_SIZE = 640
BASE_CONF = 0.65
ADAPTIVE_LAMBDA = 0.25
IOU_MATCH = 0.6
STANDARD_CONF = 0.7
CONSERVATIVE_CONF = 0.8


def yolo_to_xyxy(box):
    x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def generate_standard(model, output_dir, conf_threshold=STANDARD_CONF):
    """固定阈值伪标签"""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([
        p for p in UNLABELED_IMG_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"\n[Standard] 伪标签生成: {len(image_paths)} 张图像, 阈值={conf_threshold}")

    class_counter = defaultdict(int)
    total_boxes = 0

    for img_path in image_paths:
        result = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=conf_threshold,
            verbose=False,
        )[0]

        label_path = output_dir / f"{img_path.stem}.txt"
        if result.boxes is None or len(result.boxes) == 0:
            continue

        valid_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x, y, w, h = box.xywhn[0].tolist()
            valid_boxes.append((cls_id, x, y, w, h))
            class_counter[cls_id] += 1
            total_boxes += 1

        if valid_boxes:
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return class_counter, total_boxes


def generate_adaptive(model, output_dir, base_conf=BASE_CONF, lambda_val=ADAPTIVE_LAMBDA):
    """自适应阈值伪标签 - 根据每类AP调整阈值"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 baseline AP
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        conf=0.001,
        iou=0.6,
        augment=True,
        verbose=False,
    )
    baseline_ap = metrics.box.ap50

    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)

    class_thresholds = {}
    for cls_id, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-6)
        thres = base_conf + lambda_val * (1 - norm)
        class_thresholds[cls_id] = max(0.3, min(0.95, thres))

    print(f"\n[Adaptive] 自适应阈值: {class_thresholds}")

    image_paths = sorted([
        p for p in UNLABELED_IMG_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"[Adaptive] 伪标签生成: {len(image_paths)} 张图像")

    class_counter = defaultdict(int)
    total_boxes = 0

    for img_path in image_paths:
        result = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=0.01,
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
                class_counter[cls_id] += 1
                total_boxes += 1

        if valid_boxes:
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return class_counter, total_boxes


def generate_adaptive_consistency(model, output_dir, base_conf=BASE_CONF,
                                   lambda_val=ADAPTIVE_LAMBDA, iou_match=IOU_MATCH):
    """自适应阈值 + 翻转一致性伪标签"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 baseline AP
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        conf=0.001,
        iou=0.6,
        augment=True,
        verbose=False,
    )
    baseline_ap = metrics.box.ap50

    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)

    class_thresholds = {}
    for cls_id, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-6)
        thres = base_conf + lambda_val * (1 - norm)
        class_thresholds[cls_id] = max(0.3, min(0.95, thres))

    print(f"\n[Adaptive+Consistency] 自适应阈值: {class_thresholds}")
    print(f"[Adaptive+Consistency] IOU_MATCH={iou_match}")

    image_paths = sorted([
        p for p in UNLABELED_IMG_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"[Adaptive+Consistency] 伪标签生成: {len(image_paths)} 张图像")

    class_counter = defaultdict(int)
    total_boxes = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 原图
        res_orig = model.predict(
            source=img,
            imgsz=IMG_SIZE,
            conf=0.01,
            augment=False,
            verbose=False,
        )[0]

        # 翻转
        res_flip = model.predict(
            source=cv2.flip(img, 1),
            imgsz=IMG_SIZE,
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
            box_xyxy = yolo_to_xyxy(xywh)
            matched = False

            for fcid, fconf, fxywh in flip_boxes:
                if fcid != cid:
                    continue
                if iou_xyxy(box_xyxy, yolo_to_xyxy(fxywh)) >= iou_match:
                    matched = True
                    break

            if matched:
                x, y, w, h = xywh
                valid_boxes.append((cid, x, y, w, h))
                class_counter[cid] += 1
                total_boxes += 1

        if valid_boxes:
            label_path = output_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return class_counter, total_boxes


def main():
    parser = argparse.ArgumentParser(description="伪标签生成")
    parser.add_argument("--strategy", type=str, default="adaptive_consistency",
                        choices=["standard", "adaptive", "adaptive_consistency"],
                        help="伪标签策略")
    parser.add_argument("--conf", type=float, default=None,
                        help="固定阈值策略的置信度 (默认: 0.7)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"伪标签生成 - 策略: {args.strategy}")
    print(f"模型: {MODEL_PATH}")
    print(f"数据: {DATA_YAML}")
    print(f"Unlabeled: {UNLABELED_IMG_DIR}")
    print(f"{'='*60}")

    # 加载模型
    model = YOLO(str(MODEL_PATH))

    # 确定输出目录 (放在 pseudo_labels/{strategy}/train/ 下，与 merge 脚本匹配)
    strategy_dir_map = {
        "standard": PSEUDO_LABEL_DIR / "standard" / "train",
        "adaptive": PSEUDO_LABEL_DIR / "adaptive" / "train",
        "adaptive_consistency": PSEUDO_LABEL_DIR / "adaptive_consistency" / "train",
    }
    output_dir = strategy_dir_map[args.strategy]

    # 生成
    if args.strategy == "standard":
        conf = args.conf if args.conf is not None else STANDARD_CONF
        class_counter, total_boxes = generate_standard(model, output_dir, conf)
    elif args.strategy == "adaptive":
        class_counter, total_boxes = generate_adaptive(model, output_dir)
    else:
        class_counter, total_boxes = generate_adaptive_consistency(model, output_dir)

    # 打印统计
    print(f"\n===== 伪标签统计 ({args.strategy}) =====")
    CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for cls_id in sorted(class_counter.keys()):
        print(f"  {CLASSES[cls_id]:<20}: {class_counter[cls_id]}")
    print(f"总计: {total_boxes} 个伪标签")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
