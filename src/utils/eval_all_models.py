"""
批量评测所有模型
使用 evaluation.py 的配置: conf=0.001, iou=0.6, augment=True, split=test
"""
import sys
import csv
from pathlib import Path
from src.utils.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "experiments/thesis_model/weights"

models = [
    "baseline.pt",
    "baseline1024.pt",
    "cp640.pt",
    "cbam.pt",
    "p2_layer.pt",
    "cbam_p2_head.pt",
    "wiou.pt",
    "focal.pt",
    "data_augmentation.pt",
    "cbam_spd.pt",
    "spd_only.pt",
    "cbam_gfpn_neck.pt",
    "se.pt",
    "cbamp2.pt",
    "cbamp3.pt",
    "cbamp4.pt",
    "cbamp5.pt",
    "cbamp3+p4.pt",
]

results = []

for name in models:
    model_path = WEIGHTS_DIR / name
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print('='*60)

    try:
        res = evaluate(
            model_path=str(model_path),
            data_yaml=str(ROOT / "datasets/neu.yaml"),
            split="test",
            conf=0.001,
            iou=0.6,
            augment=True,
            verbose=True
        )

        results.append({
            "model": name.replace(".pt", ""),
            "precision": res.get("metrics/precision(B)", 0),
            "recall": res.get("metrics/recall(B)", 0),
            "mAP@0.5": res.get("metrics/mAP50(B)", 0),
            "mAP@0.5:0.95": res.get("metrics/mAP50-95(B)", 0),
        })
    except Exception as e:
        print(f"ERROR evaluating {name}: {e}")
        results.append({
            "model": name.replace(".pt", ""),
            "precision": "ERROR",
            "recall": "ERROR",
            "mAP@0.5": "ERROR",
            "mAP@0.5:0.95": "ERROR",
        })

# 打印汇总表
print("\n\n" + "="*80)
print("SUMMARY RESULTS")
print("="*80)
print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>15}")
print("-"*80)
for r in results:
    p = f"{r['precision']:.4f}" if isinstance(r['precision'], float) else r['precision']
    rec = f"{r['recall']:.4f}" if isinstance(r['recall'], float) else r['recall']
    m50 = f"{r['mAP@0.5']:.4f}" if isinstance(r['mAP@0.5'], float) else r['mAP@0.5']
    m50_95 = f"{r['mAP@0.5:0.95']:.4f}" if isinstance(r['mAP@0.5:0.95'], float) else r['mAP@0.5:0.95']
    print(f"{r['model']:<25} {p:>10} {rec:>10} {m50:>10} {m50_95:>15}")

# 保存CSV
csv_path = ROOT / "docs" / "模型评测结果汇总.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "precision", "recall", "mAP@0.5", "mAP@0.5:0.95"])
    writer.writeheader()
    writer.writerows(results)
print(f"\nCSV saved to: {csv_path}")
