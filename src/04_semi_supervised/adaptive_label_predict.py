from ultralytics import YOLO
from pathlib import Path
import numpy as np
from collections import defaultdict

def main():

    ROOT = Path(__file__).resolve().parents[2]

    MODEL_PATH = ROOT / "experiments/baseline_seed/weights/new-best.pt"
    DATA_YAML = ROOT / "datasets/neu.yaml"
    UNLABELED_IMG_DIR = ROOT / "data/NEU-DET/unlabeled-conservative/images/train"
    PSEUDO_LABEL_DIR = ROOT / "data/NEU-DET/unlabeled-conservative/pseudo_labels_adaptive/train"

    IMG_SIZE = 640
    BASE_CONF = 0.65
    LAMBDA = 0.15

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

    # ===== 生成伪标签 =====
    results = model.predict(
        source=str(UNLABELED_IMG_DIR),
        imgsz=640,
        conf=0.01,
        augment=False
    )

    # ===== 统计容器 =====
    class_counter = defaultdict(int)
    total_boxes = 0

    for result in results:
        img_path = Path(result.path)
        label_path = PSEUDO_LABEL_DIR / f"{img_path.stem}.txt"

        if result.boxes is None or len(result.boxes) == 0:
            continue

        valid_boxes = []

        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())

            thres = CLASS_CONF_THRES.get(cls_id, BASE_CONF)

            if confidence >= thres:
                x, y, w, h = box.xywhn[0].tolist()
                valid_boxes.append((cls_id, x, y, w, h))
                # 统计
                class_counter[cls_id] += 1
                total_boxes += 1

        if valid_boxes:
            with open(label_path, "w") as f:
                for cls_id, x, y, w, h in valid_boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("\n===== 伪标签统计 =====")
    for cls_id in sorted(CLASS_CONF_THRES.keys()):
        print(f"Class {cls_id}: {class_counter[cls_id]}")

    print("总伪标签数:", total_boxes)
    print("完成")


if __name__ == "__main__":
    main()