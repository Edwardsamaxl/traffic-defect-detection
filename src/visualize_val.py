from ultralytics import YOLO
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录

def visualize_val_predictions(
    model_path: Path,
    val_images_dir: Path,
    save_dir: Path
):
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    img_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
    if not img_files:
        print(f"[!] 没有找到验证集图片在 {val_images_dir}")
        return

    for img_path in img_files:
        results = model.predict(img_path, imgsz=640, conf=0.25)
        annotated_img = results[0].plot()
        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    best_model = ROOT / "experiments/baseline/weights/best.pt"
    val_dir = ROOT / "data/NEU-DET/valid/images"
    output_dir = ROOT / "experiments/baseline/val_predictions"

    visualize_val_predictions(best_model, val_dir, output_dir)
