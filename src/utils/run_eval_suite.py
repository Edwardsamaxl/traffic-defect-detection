from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalJob:
    name: str
    model_path: Path
    data_yaml: Path
    split: str = "test"
    imgsz: int = 640


def _metrics_subset(results_dict: dict[str, Any]) -> dict[str, Any]:
    # Ultralytics keys (detect):
    # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
    wanted = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    out: dict[str, Any] = {}
    for k in wanted:
        if k in results_dict:
            out[k] = float(results_dict[k])
    return out


def run_one(job: EvalJob, tta: bool, conf: float = 0.001, iou: float = 0.6) -> dict[str, Any]:
    model = YOLO(str(job.model_path))
    start = time.perf_counter()
    metrics = model.val(
        data=str(job.data_yaml),
        split=job.split,
        imgsz=job.imgsz,
        conf=conf,
        iou=iou,
        augment=tta,
        verbose=False,
    )
    elapsed_s = time.perf_counter() - start

    results_dict = dict(metrics.results_dict)
    subset = _metrics_subset(results_dict)

    return {
        "job_name": job.name,
        "model_path": str(job.model_path),
        "data_yaml": str(job.data_yaml),
        "split": job.split,
        "imgsz": job.imgsz,
        "tta": bool(tta),
        "conf": conf,
        "iou": iou,
        "elapsed_s": round(elapsed_s, 3),
        **{k: subset.get(k) for k in subset},
        "results_dict_json": json.dumps(results_dict, ensure_ascii=False),
    }


def main() -> None:
    data_yaml = ROOT / "datasets/neu.yaml"

    # 你列的对比清单（如需改文件名/策略命名，直接改这里）
    jobs: list[EvalJob] = [
        # baseline（默认参数跑出来）
        EvalJob("baseline", ROOT / "experiments/baseline_s/weights/best.pt", data_yaml, imgsz=640),

        # 最佳策略（stage4_overall best-cosine）
        EvalJob("best_strategy_stage4_cosine", ROOT / "experiments/stage4_overall/weights/best-cosine.pt", data_yaml, imgsz=640),

        # best-no-aug（你提到的“没进行数据增强的最佳策略”）
        EvalJob("best_no_aug", ROOT / "experiments/stage6_semi/weights/new-best-noaug.pt", data_yaml, imgsz=640),

        # stage4 640/1024
        EvalJob("stage4_best_640", ROOT / "experiments/stage4_overall/weights/best-640.pt", data_yaml, imgsz=640),
        EvalJob("stage4_best_1024", ROOT / "experiments/stage4_overall/weights/best-1024.pt", data_yaml, imgsz=1024),

        # baseline_seed（种子）
        EvalJob("seed_baseline", ROOT / "experiments/baseline_seed/weights/best.pt", data_yaml, imgsz=640),

        # stage6：adaptive once/twice（seed 不复制 vs 复制一遍）
        EvalJob("stage6_adaptive_once", ROOT / "experiments/stage6_semi/weights/best-adaptive-once.pt", data_yaml, imgsz=640),
        EvalJob("stage6_adaptive_twice", ROOT / "experiments/stage6_semi/weights/best-adaptive-twice.pt", data_yaml, imgsz=640),

        # stage6：new-best（conf=0.7）与 conservative（更高置信度筛选）
        EvalJob("stage6_new_best", ROOT / "experiments/stage6_semi/weights/new-best.pt", data_yaml, imgsz=640),
        EvalJob("stage6_new_best_conservative", ROOT / "experiments/stage6_semi/weights/new-best-conservative.pt", data_yaml, imgsz=640),
    ]

    # 哪些 job 需要跑 TTA on/off
    tta_matrix = {
        "baseline": [False, True],
        "best_strategy_stage4_cosine": [False, True],
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = ROOT / "reports" / f"eval_suite_{ts}.csv"
    out_md = ROOT / "reports" / f"eval_suite_{ts}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        tta_list = tta_matrix.get(job.name, [False])
        for tta in tta_list:
            if not job.model_path.exists():
                rows.append(
                    {
                        "job_name": job.name,
                        "model_path": str(job.model_path),
                        "error": "model_not_found",
                    }
                )
                continue
            rows.append(run_one(job, tta=tta))

    # write csv
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write simple markdown summary (sorted by mAP50)
    def key_map50(r: dict[str, Any]) -> float:
        v = r.get("metrics/mAP50(B)")
        return float(v) if isinstance(v, (int, float)) else -1.0

    summary_rows = [r for r in rows if "error" not in r]
    summary_rows.sort(key=key_map50, reverse=True)

    lines: list[str] = []
    lines.append("# Eval Suite Summary")
    lines.append("")
    lines.append(f"- Generated at: `{ts}`")
    lines.append(f"- Data: `{data_yaml}` split=`test`")
    lines.append(f"- CSV: `{out_csv.relative_to(ROOT)}`")
    lines.append("")
    lines.append("| rank | job | imgsz | TTA | P | R | mAP50 | mAP50-95 | elapsed(s) | model |")
    lines.append("|---:|---|---:|:---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(summary_rows, start=1):
        p = r.get("metrics/precision(B)")
        rc = r.get("metrics/recall(B)")
        m50 = r.get("metrics/mAP50(B)")
        m95 = r.get("metrics/mAP50-95(B)")
        lines.append(
            f"| {i} | {r.get('job_name')} | {r.get('imgsz')} | {str(r.get('tta'))} | "
            f"{(p if p is not None else '')} | {(rc if rc is not None else '')} | {(m50 if m50 is not None else '')} | {(m95 if m95 is not None else '')} | {r.get('elapsed_s','')} | {Path(str(r.get('model_path'))).name} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD : {out_md}")


if __name__ == "__main__":
    main()

