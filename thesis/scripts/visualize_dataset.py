"""
数据集可视化脚本 - 生成论文所需的图表
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "NEU-DET"
OUTPUT_DIR = PROJECT_ROOT / "thesis" / "figures"

# 缺陷类别
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)

# 类别颜色（RGB格式，用于OpenCV）
CLASS_COLORS = {
    0: (255, 0, 0),       # crazing - 红色
    1: (0, 255, 0),       # inclusion - 绿色
    2: (0, 0, 255),       # patches - 蓝色
    3: (255, 255, 0),     # pitted_surface - 黄色
    4: (255, 0, 255),     # rolled-in_scale - 紫色
    5: (0, 255, 255),     # scratches - 青色
}


def get_class_counts(data_yaml: Path) -> dict[str, int]:
    """统计每个类别的样本数量"""
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    data_root = data_yaml.parent.parent / data["path"]
    train_dir = data_root / data["train"].replace("images/", "").replace("../", "")

    counts = {cls: 0 for cls in CLASSES}

    for label_file in train_dir.glob("labels/*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < NUM_CLASSES:
                        counts[CLASSES[cls_id]] += 1

    return counts


def plot_data_distribution(output_path: Path):
    """绘制数据分布柱状图"""
    data_yaml = DATA_ROOT / "data.yaml"
    counts = get_class_counts(data_yaml)

    fig, ax = plt.subplots(figsize=(12, 6))

    classes = list(counts.keys())
    values = list(counts.values())
    colors = [f'#{r:02x}{g:02x}{b:02x}' for (r, g, b) in [CLASS_COLORS[i] for i in range(NUM_CLASSES)]]

    bars = ax.bar(classes, values, color=colors, edgecolor='black', linewidth=1.2)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Defect Type', fontsize=12)
    ax.set_ylabel('Number of Instances', fontsize=12)
    ax.set_title('NEU-DET Dataset: Instance Distribution by Defect Type', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Data distribution saved: {output_path}")


def get_sample_images_per_class(train_dir: Path, max_per_class: int = 2) -> dict[str, list[Path]]:
    """获取每个类别的示例图像"""
    samples = {cls: [] for cls in CLASSES}

    for label_file in sorted(train_dir.glob("labels/*.txt")):
        if len(samples[CLASSES[0]]) >= max_per_class * NUM_CLASSES:
            break

        with open(label_file) as f:
            lines = f.readlines()
            if not lines:
                continue

            # 获取第一个检测框的类别
            first_line = lines[0].strip().split()
            if len(first_line) >= 5:
                cls_id = int(first_line[0])
                if 0 <= cls_id < NUM_CLASSES and len(samples[CLASSES[cls_id]]) < max_per_class:
                    img_path = label_file.parent.parent / "images" / f"{label_file.stem}.jpg"
                    if img_path.exists():
                        samples[CLASSES[cls_id]].append(img_path)

    return samples


def draw_bounding_boxes(img: np.ndarray, label_path: Path) -> np.ndarray:
    """在图像上绘制边界框"""
    if not label_path.exists():
        return img

    with open(label_path) as f:
        lines = f.readlines()

    h, w = img.shape[:2]
    result = img.copy()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        x, y, bw, bh = map(float, parts[1:5])

        # 转换为像素坐标
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 添加类别标签
        label = f"{CLASSES[cls_id]}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return result


def plot_dataset_samples(output_path: Path):
    """绘制数据集示例图（每个类别2张）"""
    data_yaml = DATA_ROOT / "data.yaml"
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    data_root = DATA_ROOT
    train_dir = data_root / "images" / "train"

    samples = get_sample_images_per_class(train_dir, max_per_class=2)

    # 创建网格图
    n_classes = NUM_CLASSES
    n_samples = 2
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(10, 24))

    for row, cls_name in enumerate(CLASSES):
        cls_id = CLASSES.index(cls_name)
        for col, img_path in enumerate(samples[cls_name]):
            ax = axes[row, col]

            img = cv2.imread(str(img_path))
            if img is None:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 绘制边界框
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            img = draw_bounding_boxes(img, label_path)

            ax.imshow(img)
            ax.axis('off')

            if col == 0:
                color_hex = f'#{CLASS_COLORS[cls_id][0]:02x}{CLASS_COLORS[cls_id][1]:02x}{CLASS_COLORS[cls_id][2]:02x}'
                ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold', color=color_hex, loc='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dataset samples saved: {output_path}")


def visualize_adaptive_threshold():
    """生成自适应阈值流程图"""
    # 这个函数生成方法论的流程图说明
    pass


def visualize_flip_consistency():
    """生成翻转一致性筛选示意图"""
    # 这个函数生成翻转一致性筛选的示意图
    pass


def main():
    """主函数"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 绘制数据分布图
    plot_data_distribution(OUTPUT_DIR / "data_distribution.png")

    # 2. 绘制数据集示例
    plot_dataset_samples(OUTPUT_DIR / "dataset_samples.png")

    # 3. 生成训练曲线（如果有的话）
    # plot_training_curves()

    print("\n可视化完成！")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
