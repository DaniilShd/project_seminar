import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
from utils.utils import visualize_features

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SAVE_DIR = Path(cfg["paths"]["save_dir"])
PATH_VISUALIZE = Path(cfg["paths"]["path_visualize"])
ANNOTATION_FILE = Path(cfg["extract_patches"]["annotation_path"])
PATCH_DIR = Path(cfg["extract_patches"]["save_dir_patches"])
SSL_MODEL_PATH = SAVE_DIR / "dinov2_ssl_finetuned.pth"
BATCH_SIZE = (cfg["extract_features"]["batch_size"])
TARGET_SIZE = (cfg["extract_features"]["target_size"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class PatchFeatureDataset(Dataset):
    def __init__(self, ann_path, patches_dir, transform):
        with open(ann_path) as f:
            self.data = json.load(f)
        self.names = list(self.data.keys())
        self.dir = Path(patches_dir)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(self.dir / name).convert("RGB")
        label = self.data[name]["class"] - 1
        return self.transform(img), label


if __name__ == "__main__":
    print("Loading SSL-finetuned DINO...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.load_state_dict(torch.load(SSL_MODEL_PATH)['student_state_dict'])
    model.to(DEVICE).eval()

    transform = T.Compose([
        T.Resize((TARGET_SIZE,TARGET_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = PatchFeatureDataset(ANNOTATION_FILE, PATCH_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    features, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting SSL features"):
            feats = model(imgs.to(DEVICE))
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    np.save(SAVE_DIR / "features_ssl.npy", features)
    np.save(SAVE_DIR / "labels_ssl.npy", labels)

    visualize_features(features, labels, PATH_VISUALIZE / "ssl_features_pca.png")

    print("SSL feature extraction complete!")
    print(f"Features shape: {features.shape}")