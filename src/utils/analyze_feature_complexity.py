import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===============================
# 项目路径
# ===============================
ROOT = Path(__file__).resolve().parents[2]


# ===============================
# 1️⃣ Feature Extraction (YOLOv8专用)
# ===============================
def extract_features(model, dataloader, device):

    all_features = []
    all_labels = []

    backbone = model.model.model[:22]  # 去掉 Detect
    backbone.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):

            imgs = batch["img"].to(device)
            cls = batch["cls"].cpu().numpy()

            feats = backbone(imgs)

            # 如果输出是 list（多尺度），取最后一层
            if isinstance(feats, list):
                feats = feats[-1]

            # Global Average Pooling
            feats = torch.mean(feats, dim=[2, 3])
            feats = feats.cpu().numpy()

            all_features.append(feats)
            all_labels.append(cls)

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


# ===============================
# 2️⃣ Complexity Metrics
# ===============================
def compute_complexity(features, labels):

    classes = np.unique(labels)

    intra_vars = []
    class_means = []

    for c in classes:
        feat_c = features[labels == c]
        mean_c = np.mean(feat_c, axis=0)
        class_means.append(mean_c)

        var_c = np.mean(np.sum((feat_c - mean_c) ** 2, axis=1))
        intra_vars.append(var_c)

    intra_variance = np.mean(intra_vars)

    class_means = np.vstack(class_means)
    inter_dist = pairwise_distances(class_means)
    inter_dist = inter_dist[np.triu_indices_from(inter_dist, k=1)]
    inter_distance = np.mean(inter_dist)

    fisher_ratio = inter_distance / intra_variance

    print("===== Data Complexity Report =====")
    print(f"Intra-class Variance : {intra_variance:.4f}")
    print(f"Inter-class Distance : {inter_distance:.4f}")
    print(f"Fisher Ratio         : {fisher_ratio:.4f}")

    return intra_variance, inter_distance, fisher_ratio


# ===============================
# 3️⃣ t-SNE Visualization
# ===============================
def visualize_tsne(features, labels):

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(6, 6))

    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            label=f"class {c}",
            alpha=0.6
        )

    plt.legend()
    plt.title("Feature Space Visualization")
    plt.show()


# ===============================
# Main
# ===============================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = ROOT / "experiments/stage4_overall/weights/best-cosine.pt"
    data_yaml = ROOT / "datasets/neu.yaml"

    model = YOLO(str(model_path))
    model.model.eval()

    features = []
    labels = []

    results = model.predict(
        source=str(ROOT / "data/NEU-DET/images/val"),
        imgsz=640,
        batch=16,
        stream=True,
        verbose=False
    )

    with torch.no_grad():
        for r in results:

            img = r.orig_img
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = img.unsqueeze(0).to(device) / 255.0

            # 关键：直接走模型 forward
            outputs = model.model(img)

            # outputs 是 list，多尺度特征
            # 取最后一个尺度特征
            feat = outputs[0] if isinstance(outputs, tuple) else outputs

            if isinstance(feat, list):
                feat = feat[-1]

            feat = torch.mean(feat, dim=[2, 3])
            features.append(feat.cpu().numpy())

            if len(r.boxes.cls) > 0:
                labels.append(r.boxes.cls.cpu().numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    compute_complexity(features, labels)
    visualize_tsne(features, labels)