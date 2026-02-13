from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    data_yaml = ROOT / "datasets/neu.yaml"

    stage1_weights = (
        ROOT / "experiments/stage1_data_aug/weights/best.pt"
    )

    project_path = ROOT / "experiments"

    model = YOLO(str(stage1_weights))

    # ========== Stage 2: Fine-tuning ==========
    model.train(
        data=str(data_yaml),

        epochs=40,

        lr0=0.001,
        lrf=0.01,

        optimizer="SGD",

        batch=4,
        imgsz=640,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        patience=20,
        workers=2,
        device=0,
        amp=True,

        project=str(project_path),
        name="stage2_finetune",
    )

if __name__ == "__main__":
    main()
