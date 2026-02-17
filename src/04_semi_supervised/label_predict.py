from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ================= 配置区 =================
MODEL_PATH = ROOT / "experiments/baseline_seed/weights/new-best.pt"

UNLABELED_IMG_DIR = ROOT / "data/NEU-DET/unlabeled/images/train"

PSEUDO_LABEL_DIR = ROOT / "data/NEU-DET/unlabeled/pseudo_labels/train"

CONF_THRES = 0.7
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

for result in results:
    img_path = Path(result.path)
    label_path = PSEUDO_LABEL_DIR / f"{img_path.stem}.txt"

    if result.boxes is None or len(result.boxes) == 0:
        continue

    with open(label_path, "w") as f:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x, y, w, h = box.xywhn[0].tolist()
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("伪标签生成完成")
