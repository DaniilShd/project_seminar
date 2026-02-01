import torch
import urllib.request
from pathlib import Path

# Ссылка для скачивания
url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
save_path = Path("dinov2_vits14_pretrain.pth")

print(f"Downloading DINOv2 weights...")
urllib.request.urlretrieve(url, save_path)
print(f"Weights saved to: {save_path}")