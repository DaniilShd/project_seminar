import json
import torch
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from utils.utils import visualize_features


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

with open("config.yaml", "r") as f:
    cfg = json.load(f) if False else __import__('yaml').safe_load(f)  # если YAML

PATCH_DIR = Path(cfg["extract_patches"]["save_dir_patches"])
ANNOTATION_FILE = Path(cfg["extract_patches"]["annotation_path"])
SAVE_DIR = Path(cfg["paths"]["save_dir"])
VIS_DIR = Path(cfg["paths"]["path_visualize"])
BATCH_SIZE = (cfg["extract_features"]["batch_size"])

TARGET_SIZE = (cfg["image_params"]["target_width"], cfg["image_params"]["target_height"])
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


class PatchDataset(Dataset):
    def __init__(self, annotation_file, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.patch_names = list(self.annotations.keys())
        self.transform = transform

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        info = self.annotations[patch_name]
        patch_path = PATCH_DIR / patch_name
        patch = Image.open(patch_path).convert('RGB')
        label = info["class"] - 1  # вычитаем 1, как в предыдущем скрипте

        if self.transform:
            patch = self.transform(patch)
        return patch, label


patch_transform = T.Compose([
    T.Resize(TARGET_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_dinov2(model_size='small'):
    print(f"Loading DINOv2 {model_size}...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    feat_dim = 384
    model.to(DEVICE)
    model.eval()
    print(f"DINOv2 loaded. Feature dimension: {feat_dim}")
    return model, feat_dim


def extract_features(dataset, model, batch_size=32, device=None):
    if device is None:
        device = DEVICE

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features = []
    all_labels = []

    model.eval()
    model.to(device)

    for batch, lbl in tqdm(loader, desc="Extracting features"):
        batch = batch.to(device)
        with torch.no_grad():
            feats = model(batch)
        all_features.append(feats.cpu().numpy())
        all_labels.append(lbl.numpy())
        del batch, feats

    if all_features:
        return np.vstack(all_features), np.concatenate(all_labels)
    else:
        return np.array([]), np.array([])



if __name__ == "__main__":
    dataset = PatchDataset(ANNOTATION_FILE, transform=patch_transform)
    print(f"Total patches: {len(dataset)}")

    model, feat_dim = load_dinov2('small')

    features, labels = extract_features(dataset, model, batch_size=16)

    # Сохраняем PCA визуализацию
    visualize_features(features, labels, VIS_DIR / "features_pca.png")

    # Сохраняем фичи и метки
    np.save(SAVE_DIR / "features.npy", features)
    np.save(SAVE_DIR / "labels.npy", labels)

    # Сохраняем статистику классов
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_info = {
        "num_classes": len(unique_labels),
        "class_distribution": {str(int(lbl)): int(cnt) for lbl, cnt in zip(unique_labels, counts)},
        "feature_dim": int(feat_dim),
        "num_patches": len(features)
    }
    with open(SAVE_DIR / "class_info.json", 'w') as f:
        json.dump(class_info, f, indent=2)

    print(f"Features saved to: {SAVE_DIR / 'features.npy'}")
    print(f"Labels saved to: {SAVE_DIR / 'labels.npy'}")
    print(f"Class info saved to: {SAVE_DIR / 'class_info.json'}")
