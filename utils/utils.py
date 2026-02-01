import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_features(features, labels, save_path):
    # Ограничение до 500 точек для визуализации
    if len(features) > 500:
        idx = np.random.choice(len(features), 500, replace=False)
        features = features[idx]
        labels = labels[idx]

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
    plt.colorbar(label='Class')
    plt.title(f"PCA of Features (Points: {len(features)})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    return pca