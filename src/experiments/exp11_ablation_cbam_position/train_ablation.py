"""
Exp-11: CBAM 集成位置消融实验
=====================================
实验目标：验证 CBAM 插入在不同 Backbone 层级对检测性能的影响

用法：
  # 单实验模式（Kaggle 并行训练）
  python train_ablation.py --exp exp11_baseline --yaml yolov8s.yaml
  python train_ablation.py --exp exp11_cbam_p3only --yaml yolov8s_cbam_p3only.yaml

  # 批量模式（本地顺序训练）
  python train_ablation.py

实验设计：
  - Baseline:     纯 YOLOv8s（无 CBAM）
  - CBAM-P3:      仅在 P3/8 层后插入 CBAM
  - CBAM-P4:      仅在 P4/16 层后插入 CBAM
  - CBAM-P5:      仅在 P5/32 层后插入 CBAM
  - CBAM-P3+P4:   在 P3/8 和 P4/16 层后插入 CBAM（共2个）

注意：CBAM-Full(3) = 02_cbam (P3+P4+P5, mAP50=0.7870) 已训好，作为参考不重训

数据集：NEU-DET (640x640)
评价指标：Precision, Recall, mAP@0.5, mAP@0.5:0.95
"""
from pathlib import Path
from ultralytics import YOLO
import json
import time
import argparse

ROOT = Path(__file__).parent.parent.parent.parent
NEU_DATA = ROOT / "data/NEU-DET"
PRETRAINED = ROOT / "yolov8s.pt"


def create_yaml():
    yaml_path = ROOT / "datasets/neu_ablation.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {NEU_DATA}
train: images/train
val: images/val
test: images/test

names:
 0: crazing
 1: inclusion
 2: patches
 3: pitted_surface
 4: rolled-in_scale
 5: scratches
""")
    return yaml_path


TRAIN_CONFIG = {
    "epochs": 200,
    "patience": 50,
    "imgsz": 640,
    "batch": 4,
    "workers": 4,
    "project": str(ROOT / "experiments"),
    "pretrained": str(PRETRAINED),
    "optimizer": "auto",
    "verbose": True,
    "mosaic": 1.0,
    "mixup": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}


def train_and_eval(exp_name: str, model_yaml: str) -> dict:
    print(f"\n{'='*60}")
    print(f"开始实验: {exp_name}")
    print(f"模型配置: {model_yaml}")
    print(f"{'='*60}")

    yaml_path = create_yaml()
    cfg = dict(TRAIN_CONFIG)
    cfg["data"] = str(yaml_path)
    cfg["name"] = exp_name

    model = YOLO(model_yaml)
    model.train(**cfg)
    metrics = model.val()

    return {
        "model_name": exp_name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def main_single(exp_name: str, model_yaml: str):
    model_yaml = str(ROOT / model_yaml)
    result = train_and_eval(exp_name, model_yaml)
    print(f"\n最终结果: {exp_name}")
    print(f"  P: {result['precision']:.4f}  R: {result['recall']:.4f}  "
          f"mAP50: {result['mAP50']:.4f}  mAP50-95: {result['mAP50-95']:.4f}")
    results_file = ROOT / f"experiments/{exp_name}_result.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已保存: {results_file}")


def main_batch():
    experiments = {
        "exp11_baseline":     "ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml",
        "exp11_cbam_p3only":  "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p3only.yaml",
        "exp11_cbam_p4only":  "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p4only.yaml",
        "exp11_cbam_p5only":  "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p5only.yaml",
        "exp11_cbam_p3p4":    "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p3p4.yaml",
    }

    results = []
    start_time = time.time()

    for exp_name, model_yaml in experiments.items():
        try:
            result = train_and_eval(exp_name, str(ROOT / model_yaml))
            results.append(result)
            print(f"\n结果: {exp_name}")
            print(f"  P: {result['precision']:.4f}  R: {result['recall']:.4f}  "
                  f"mAP50: {result['mAP50']:.4f}  mAP50-95: {result['mAP50-95']:.4f}")
        except Exception as e:
            print(f"实验 {exp_name} 失败: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"消融实验完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    print(f"\n{'模型':<20} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["mAP50"], reverse=True):
        print(f"{r['model_name']:<20} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f}")

    results_file = ROOT / "experiments/exp11_ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="CBAM 位置消融实验")
    parser.add_argument("--exp", type=str, help="实验名")
    parser.add_argument("--yaml", type=str, help="模型yaml路径")
    args = parser.parse_args()

    if args.exp and args.yaml:
        main_single(args.exp, args.yaml)
    else:
        main_batch()


if __name__ == "__main__":
    main()
