from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


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

def main():

    ROOT = Path(__file__).resolve().parents[2]

    MODEL_PATH = ROOT / "experiments/baseline_seed/weights/new-best.pt"
    DATA_YAML = ROOT / "datasets/neu.yaml"
    UNLABELED_IMG_DIR = ROOT / "data/NEU-DET/unlabeled-conservative/images/train"
    PSEUDO_LABEL_DIR = ROOT / "data/NEU-DET/unlabeled-conservative/pseudo_labels_adaptive_consistency/train"

    IMG_SIZE = 640
    BASE_CONF = 0.65
    LAMBDA = 0.25
    IOU_MATCH = 0.6

    PSEUDO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    # ===== 自动获取 baseline AP =====
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        conf=0.001,
        iou=0.6,
        augment=True,
        verbose=False
    )

    baseline_ap = metrics.box.ap50

    ap_min = min(baseline_ap)
    ap_max = max(baseline_ap)

    CLASS_CONF_THRES = {}

    for cls_id, ap in enumerate(baseline_ap):
        norm = (ap - ap_min) / (ap_max - ap_min + 1e-6)
        thres = BASE_CONF + LAMBDA * (1 - norm)
        CLASS_CONF_THRES[cls_id] = round(float(thres), 3)

    print("自动阈值:", CLASS_CONF_THRES)

    # ===== 统计容器 =====
    class_counter = defaultdict(int)
    total_boxes = 0
    total_images = 0

    image_paths = sorted(
        [p for p in UNLABELED_IMG_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    for img_path in image_paths:
        total_images += 1
        label_path = PSEUDO_LABEL_DIR / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res_orig = model.predict(source=img, imgsz=IMG_SIZE, conf=0.01, augment=False, verbose=False)[0]
        res_flip = model.predict(source=img[:, ::-1], imgsz=IMG_SIZE, conf=0.01, augment=False, verbose=False)[0]

        if res_orig.boxes is None or len(res_orig.boxes) == 0:
            continue

        orig_boxes = []
        for b in res_orig.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            th = CLASS_CONF_THRES.get(cid, BASE_CONF)
            if conf >= th:
                xywh = b.xywhn[0].tolist()
                orig_boxes.append((cid, conf, xywh))

        flip_boxes = []
        if res_flip.boxes is not None and len(res_flip.boxes) > 0:
            for b in res_flip.boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                th = CLASS_CONF_THRES.get(cid, BASE_CONF)
                if conf >= th:
                    x, y, w, h = b.xywhn[0].tolist()
                    x = 1.0 - x
                    flip_boxes.append((cid, conf, [x, y, w, h]))

        valid_boxes = []
        for cid, conf, xywh in orig_boxes:
            box_xyxy = yolo_to_xyxy(xywh)
            matched = False
            for fcid, fconf, fxywh in flip_boxes:
                if fcid != cid:
                    continue
                if iou_xyxy(box_xyxy, yolo_to_xyxy(fxywh)) >= IOU_MATCH:
                    matched = True
                    conf = min(conf, fconf)
                    break
            if matched:
                x, y, w, h = xywh
                valid_boxes.append((cid, x, y, w, h))
                class_counter[cid] += 1
                total_boxes += 1

        if valid_boxes:
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("\n===== 伪标签统计 =====")
    for cls_id in sorted(CLASS_CONF_THRES.keys()):
        print(f"Class {cls_id}: {class_counter[cls_id]}")

    print("总伪标签数:", total_boxes)
    print("推理图像数:", total_images)
    print("完成")


if __name__ == "__main__":
    main()