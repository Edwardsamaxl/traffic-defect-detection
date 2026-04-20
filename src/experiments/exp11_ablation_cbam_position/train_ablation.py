"""
Exp-11: CBAM 集成位置消融实验
=====================================
实验目标：验证 CBAM 插入在不同 Backbone 层级对检测性能的影响

⚠️ 命名说明：
  - Backbone 里的 P2/4、P3/8、P4/16、P5/32 是降采样阶段标记
  - 例如 "P3/8" 意思是 stride=8 的特征输出
  - 检测层 P3/P4/P5 是 head 的输出尺度，与 backbone 阶段不是同一套命名

CBAM 插入位置 = backbone 降采样阶段之后插入 CBAM 模块

用法：
  # 单实验模式（Kaggle 并行训练）
  python train_ablation.py --exp exp11_cbam_p2only --yaml yolov8s_cbam_p2only.yaml
  python train_ablation.py --exp exp11_cbam_p2p3 --yaml yolov8s_cbam_p2p3.yaml

  # 批量模式（本地顺序训练）
  python train_ablation.py

实验设计（按 backbone 降采样阶段分类）：
  - exp11_baseline:     纯 YOLOv8s（无 CBAM）-------------------- 公平对照
  - exp11_cbam_p2only: 仅在 P2/4 阶段后插入 CBAM（160×160）----- 细粒度
  - exp11_cbam_p3only: 仅在 P3/8 阶段后插入 CBAM（80×80）
  - exp11_cbam_p4only: 仅在 P4/16 阶段后插入 CBAM（40×40）
  - exp11_cbam_p5only: 仅在 P5/32 阶段后插入 CBAM（20×20）
  - exp11_cbam_p2p3:   在 P2/4 + P3/8 阶段后插入 CBAM（细粒度组合）
  - exp11_cbam_p3p4:   在 P3/8 + P4/16 阶段后插入 CBAM（浅中组合）
  - exp11_cbam_full:    在 P2/4 + P3/8 + P4/16 阶段后插入 CBAM（=exp02_cbam，已训好）

注意：exp11_cbam_full (P2+P3+P4) 即 exp02_cbam (mAP50=0.7870)，已训好不重训

数据集：NEU-DET (640x640)
评价指标：Precision, Recall, mAP@0.5, mAP@0.5:0.95
"""
from pathlib import Path
from ultralytics import YOLO
import json
import time
import argparse
import urllib.request

# 动态计算 ROOT：自动向上找到包含 ultralytics-main 的目录
_script = Path(__file__).resolve()
for _depth in range(2, 6):
    _root = _script
    for _ in range(_depth):
        _root = _root.parent
    if (_root / "ultralytics-main").exists():
        break
ROOT = _root
NEU_DATA = ROOT / "data/NEU-DET"
PRETRAINED = ROOT / "yolov8s.pt"

# 调试：确认路径
print(f"[DEBUG] ROOT = {ROOT}")
print(f"[DEBUG] PRETRAINED = {PRETRAINED}  exists={PRETRAINED.exists()}")

# 自动下载预训练权重（如果不存在）
if not PRETRAINED.exists():
    print(f"下载预训练权重到 {PRETRAINED} ...")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    urllib.request.urlretrieve(url, str(PRETRAINED))
    print("下载完成")


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
    "amp": False,
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
    print(f"ROOT: {ROOT}")
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
    # 实验配置：(实验名, yaml路径, 是否跳过[已有结果或已知已训练])
    # skip=True 表示该实验已训好（如exp02_cbam），跳过不重训
    experiments = [
        ("exp11_baseline",    "ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml",             False),
        ("exp11_cbam_p2only", "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p2only.yaml",  False),  # 新增
        ("exp11_cbam_p3only", "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p3only.yaml",  False),
        ("exp11_cbam_p4only", "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p4only.yaml",  False),
        ("exp11_cbam_p5only", "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p5only.yaml",  False),
        ("exp11_cbam_p2p3",   "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p2p3.yaml",    False),   # 新增
        ("exp11_cbam_p3p4",   "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam_p3p4.yaml",    False),
        # exp11_cbam_full = exp02_cbam (P2+P3+P4, mAP50=0.7870)，已训好，跳过
        ("exp11_cbam_full",    "ultralytics-main/ultralytics/cfg/models/v8/yolov8s_cbam.yaml",         True),
    ]

    results = []
    start_time = time.time()

    for exp_name, model_yaml, skip in experiments:
        # 检查是否已有结果文件（跳过已完成的实验，避免重复训练）
        result_file = ROOT / f"experiments/{exp_name}_result.json"
        if result_file.exists():
            print(f"\n[跳过] {exp_name} 已存在结果文件: {result_file}，跳过训练")
            with open(result_file, "r") as f:
                results.append(json.load(f))
            continue

        if skip:
            print(f"\n[跳过] {exp_name} 已知已训好（如exp02_cbam），跳过训练")
            continue

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
