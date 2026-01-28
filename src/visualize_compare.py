import matplotlib.pyplot as plt
import numpy as np

# 类别
classes = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

# baseline 指标（来自你的 baseline 结果）
baseline_map50 = [0.609, 0.740, 0.827, 0.995, 0.736, 0.879]

# stage1 指标（来自 stage1 结果）
stage1_map50 = [0.701, 0.862, 0.893, 0.995, 0.592, 0.796]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, baseline_map50, width, label="Baseline")
plt.bar(x + width/2, stage1_map50, width, label="Stage 1")

plt.xticks(x, classes, rotation=30)
plt.ylabel("mAP@0.5")
plt.title("Per-class mAP@0.5 Comparison")
plt.legend()
plt.tight_layout()
plt.show()
