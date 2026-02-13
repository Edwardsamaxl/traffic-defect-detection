from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
MODEL_PATH = ROOT / "experiments/baseline_seed/weights/best.pt"

UNLABELED_IMG_DIR = ROOT / "data/NEU-DET-semi/unlabeled/images/train"

PSEUDO_LABEL_DIR = ROOT / "data/NEU-DET-semi/unlabeled/pseudo_labels-conservative/train"

CONF_THRES = 0.75
IMG_SIZE = 640
# =========================================

PSEUDO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(UNLABELED_IMG_DIR),
    imgsz=IMG_SIZE,
    conf=CONF_THRES,
    save=False,
    stream=True
)

total_imgs = 0
total_boxes = 0

for result in results:
    total_imgs += 1
    img_path = Path(result.path)
    label_path = PSEUDO_LABEL_DIR / f"{img_path.stem}.txt"

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        label_path.touch()
        continue

    with open(label_path, "w") as f:
        for box in boxes:
            cls_id = int(box.cls.item())
            x, y, w, h = box.xywhn[0].tolist()
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            total_boxes += 1

print("===== Pseudo-label 生成完成 =====")
print(f"Images processed : {total_imgs}")
print(f"Total boxes kept : {total_boxes}")
print(f"Avg boxes/image  : {total_boxes / max(total_imgs,1):.3f}")
print(f"Confidence thres : {CONF_THRES}")
