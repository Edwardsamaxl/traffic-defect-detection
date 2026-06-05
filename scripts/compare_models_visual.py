"""
模型检测效果可视化对比脚本
用法: python scripts/compare_models_visual.py --model1 experiments/thesis_model/weights/baseline.pt --model2 experiments/thesis_model/weights/cbam.pt --num 10
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path("E:/PycharmProjects/traffic-defect-detection")
TEST_DIR = ROOT / "data/NEU-DET/images/test"
DEFAULT_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def put_text_chinese(img: np.ndarray, text: str, pos: tuple, color=(0, 0, 255), fontsize=28):
    """在OpenCV图像上绘制中文"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def plot_result(result, model_name: str) -> np.ndarray:
    """用 ultralytics 自带 plot 并标注模型名"""
    plotted = result.plot(line_width=2, font_size=0.6)
    h, w = plotted.shape[:2]
    # 顶部加黑条写模型名
    bar_h = 40
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:, :] = plotted
    canvas = put_text_chinese(canvas, model_name, (10, 4), color=(255, 255, 255), fontsize=26)
    return canvas


def compare_two_models(model1_path: str, model2_path: str, num_images: int = 10, conf: float = 0.25, imgsz: int = 640, seed: int = 42):
    out_dir = ROOT / f"runs/compare/{Path(model1_path).stem}_vs_{Path(model2_path).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型1: {model1_path}")
    m1 = YOLO(model1_path)
    print(f"加载模型2: {model2_path}")
    m2 = YOLO(model2_path)

    # 选图
    images = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png")) + sorted(TEST_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(f"在 {TEST_DIR} 没找到图片")

    random.seed(seed)
    selected = random.sample(images, min(num_images, len(images)))
    print(f"共 {len(images)} 张测试图，随机选 {len(selected)} 张进行对比，结果保存到 {out_dir}")

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 推理
        r1 = m1(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        r2 = m2(img, conf=conf, imgsz=imgsz, verbose=False)[0]

        p1 = plot_result(r1, f"模型1: {Path(model1_path).stem}")
        p2 = plot_result(r2, f"模型2: {Path(model2_path).stem}")

        # 统一高度
        target_h = max(img.shape[0], p1.shape[0], p2.shape[0])
        target_h = max(target_h, 640)

        def resize_to_h(im, h):
            scale = h / im.shape[0]
            return cv2.resize(im, (int(im.shape[1] * scale), h))

        img_s = resize_to_h(img, target_h)
        p1_s = resize_to_h(p1, target_h)
        p2_s = resize_to_h(p2, target_h)

        # 拼接
        combined = np.hstack([img_s, p1_s, p2_s])

        # 底部标注文件名
        combined = put_text_chinese(combined, f"图片: {img_path.name}", (10, combined.shape[0] - 10), color=(0, 255, 0), fontsize=24)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), combined)
        d1 = len(r1.boxes) if r1.boxes is not None else 0
        d2 = len(r2.boxes) if r2.boxes is not None else 0
        print(f"  {img_path.name}: 模型1检出{d1}个, 模型2检出{d2}个 -> {out_path.name}")

    print(f"\n对比完成，结果保存在: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="对比两个YOLO模型的真实检测效果")
    parser.add_argument("--model1", required=True, help="模型1权重路径")
    parser.add_argument("--model2", required=True, help="模型2权重路径")
    parser.add_argument("--num", type=int, default=10, help="对比图片数量（默认10）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认0.25）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸（默认640）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    compare_two_models(args.model1, args.model2, args.num, args.conf, args.imgsz, args.seed)


if __name__ == "__main__":
    main()
