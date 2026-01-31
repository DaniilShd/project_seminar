import json
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ANNOTATION_PATH = Path(cfg["paths"]["annotation_path"])
TARGET_SIZE = (cfg["image_params"]["target_width"], cfg["image_params"]["target_height"])
MAX_PATCHES_PER_IMG = cfg["training"].get("max_patches_per_img", 5)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path(cfg["paths"]["save_dir"])


class PatchDataset(Dataset):
    """Датасет, который на лету создаёт патчи из аннотаций."""
    
    def __init__(self, annotations, transform=None, max_patches=5, context=0.2):
        self.transform = transform
        self.context = context
        self.max_patches = max_patches
        self.samples = []

        for img_id, data in annotations.items():
            img_path = Path(data["image_path"])
            bboxes = data["bboxes"][:max_patches]
            classes = data["classes"][:max_patches]

            for bbox, cls in zip(bboxes, classes):
                self.samples.append({
                    "img_path": img_path,
                    "bbox": bbox,
                    "class": cls
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patch = self.extract_patch(sample["img_path"], sample["bbox"])
        label = sample["class"] - 1  # Сдвиг классов

        if self.transform:
            patch = self.transform(patch)

        return patch, label

    def extract_patch(self, img_path, bbox):
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = bbox

        # Центральный квадрат с контекстом
        w, h = x2 - x1, y2 - y1
        square_side = max(w, h)
        square_side = int(square_side * (1 + 2 * self.context))
        square_side = max(16, min(square_side, img.width, img.height))

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sx1 = max(0, cx - square_side // 2)
        sy1 = max(0, cy - square_side // 2)
        sx2 = min(img.width, sx1 + square_side)
        sy2 = min(img.height, sy1 + square_side)

        patch = img.crop((sx1, sy1, sx2, sy2))
        patch = patch.resize(TARGET_SIZE)
        return patch


def load_dinov2(model_size="small"):
    print(f"Loading DINOv2 {model_size}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(DEVICE)
    model.eval()
    print("DINOv2 loaded.")
    return model

def extract_features(dataset, model, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features, all_labels = [], []

    with torch.no_grad():
        for batch, labels in tqdm(loader, desc="Extracting features"):
            batch = batch.to(DEVICE)
            feats = model(batch)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            del batch, feats

    if all_features:
        return np.vstack(all_features), np.concatenate(all_labels)
    return np.array([]), np.array([])

def visualize_patches(dataset, save_path="patches.png", num=12):
    idx = np.random.choice(len(dataset), min(num, len(dataset)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i < len(idx):
            patch, label = dataset[idx[i]]
            if torch.is_tensor(patch):
                patch = patch.permute(1, 2, 0).numpy()  # CHW -> HWC
            ax.imshow(patch)
            ax.set_title(f"Class {label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_features(features, labels, save_path="pca.png"):
    if len(features) > 500:
        idx = np.random.choice(len(features), 500, replace=False)
        features, labels = features[idx], labels[idx]

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=50)
    plt.colorbar(label="Class")
    plt.title("PCA of DINOv2 Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return pca


if __name__ == "__main__":
    with open(ANNOTATION_PATH, "r") as f:
        data = json.load(f)

    # Подготовка аннотаций
    annotations = {
        img_id.replace(".jpg", ""): {
            "image_path": info["image_path"],
            "bboxes": info["bboxes"],
            "classes": info["classes"]
        } for img_id, info in data.items()
    }

    # Трансформации для патчей
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PatchDataset(annotations, transform=transform, max_patches=MAX_PATCHES_PER_IMG)

    # Визуализация патчей
    visualize_patches(dataset, "sample_patches.png", num=12)

    # Загрузка модели
    model = load_dinov2()

    # Извлечение фичей
    features, labels = extract_features(dataset, model, batch_size=16)

    # PCA визуализация
    visualize_features(features, labels, save_path="features_pca.png")





import json
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# Параметры (меняй при необходимости)
IMAGE_DIR = "./data/images/"         # Папка с изображениями

MAX_IMAGES = None             # None = все изображения, или укажи число для ограничения
ANNOTATION_PATH = "./dataset/annotations.json"
MAX_PATCHES_PER_IMG = 5       # Макс дефектов с одного изображения
SAVE_DIR = "res/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Размеры изображений (указаны в задании)
IMG_WIDTH = 1600
IMG_HEIGHT = 256

# Целевой размер для модели
TARGET_SIZE = (518, 518)


with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

annotations = {}

for img_id, info in data.items():
    annotations[img_id.replace('.jpg', '')] = {
        'image_path': info['image_path'],
        'bboxes': info['bboxes'],
        'classes': info['classes']
    }

    # Статистика
total_bboxes = sum(len(v['bboxes']) for v in annotations.values())
class_counts = {}
for v in annotations.values():
    for cls in v['classes']:
        class_counts[cls] = class_counts.get(cls, 0) + 1

print(f"Total bounding boxes: {total_bboxes}")
print(f"Unique classes: {len(class_counts)}")
print(f"Class distribution: {class_counts}")



def extract_circumscribed_squares(image_path, bboxes, class_ids, target_size=(518,518), context=0.2):
    """Исправленная версия для больших дефектов"""
    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size
    patches, labels, info = [], [], []
    
    for bbox, cls in zip(bboxes, class_ids):
        x1, y1, x2, y2 = bbox
        
        # Проверь размер дефекта
        defect_w = x2 - x1
        defect_h = y2 - y1
        
        # Если дефект слишком широкий (почти во всю ширину)
        if defect_w > img_w * 0.8:  # больше 80% ширины
            # Берем фиксированный размер квадрата
            square_side = min(448, img_h)  # максимум 448 или высота изображения
        else:
            # Обычная логика
            square_side = max(defect_w, defect_h)
        
        # Добавляем контекст (но не для огромных дефектов)
        if defect_w < img_w * 0.7:  # только для нешироких дефектов
            square_side = int(square_side * (1 + 2 * context))
        
        # Ограничиваем
        square_side = min(square_side, img_w, img_h, 448)
        square_side = max(square_side, 16)
        
        # Центр
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Координаты квадрата
        sx1 = center_x - square_side // 2
        sy1 = center_y - square_side // 2
        sx2 = sx1 + square_side
        sy2 = sy1 + square_side
        
        # Корректировка
        if sx1 < 0: sx1, sx2 = 0, square_side
        if sy1 < 0: sy1, sy2 = 0, square_side
        if sx2 > img_w: sx1, sx2 = img_w - square_side, img_w
        if sy2 > img_h: sy1, sy2 = img_h - square_side, img_h
        
        sx1, sy1, sx2, sy2 = int(sx1), int(sy1), int(sx2), int(sy2)
        
        if sx2 > sx1 and sy2 > sy1:
            patch = img.crop((sx1, sy1, sx2, sy2))
            patch = patch.resize(target_size)
            
            # Проверь, не черный ли
            patch_np = np.array(patch)
            if patch_np.mean() > 10:  # не черный
                patches.append(patch)
                # ИСПРАВЛЕНИЕ: вычитаем 1
                labels.append(cls - 1)  # ← ЗДЕСЬ ВЫЧИТАЕМ 1!
                info.append({
                    'original_bbox': (x1, y1, x2, y2),
                    'square_bbox': (sx1, sy1, sx2, sy2),
                    'square_size': square_side,
                    'class': cls  # Оригинальный класс сохраняем
                })
    
    return patches, labels, info
def visualize_extraction_process(img_path, bbox, patch_info):
    """Визуализирует процесс извлечения патча"""
    from PIL import Image
    
    img = Image.open(img_path).convert('RGB')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. Исходное изображение с bbox
    axes[0].imshow(img)
    x1, y1, x2, y2 = bbox
    axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2))
    axes[0].set_title(f'Original bbox\nSize: {x2-x1}x{y2-y1}')
    
    # 2. Описанный квадрат
    axes[1].imshow(img)
    sq_x1, sq_y1, sq_x2, sq_y2 = patch_info['square_bbox']
    axes[1].add_patch(plt.Rectangle((sq_x1, sq_y1), sq_x2-sq_x1, sq_y2-sq_y1,
                                   fill=False, edgecolor='blue', linewidth=2))
    axes[1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='red', linewidth=2))
    axes[1].set_title(f'Circumscribed square\nSize: {patch_info["square_size"]}px')
    
    # 3. Финальный патч
    patch = img.crop((sq_x1, sq_y1, sq_x2, sq_y2))
    patch = patch.resize((518, 518))
    axes[2].imshow(patch)
    axes[2].set_title(f'Final patch\nClass: {patch_info["class"]}')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()



class PatchDataset(Dataset):
    def __init__(self, annotations, transform=None, max_patches=5, 
                 context=0.2, target_size=(TARGET_SIZE), visualize_first=True):
        self.transform = transform
        self.target_size = target_size
        self.context = context
        self.max_patches = max_patches
        
        # Вместо хранения патчей, храним только информацию для их создания
        self.samples = []
        
        visualized = False
        
        for img_id, data in annotations.items():
            img_path = data['image_path']
            bboxes = data['bboxes'][:max_patches]
            classes = data['classes'][:max_patches]
            
            if not bboxes:
                continue
            
            # Сохраняем информацию для создания патчей
            for bbox, cls in zip(bboxes, classes):
                self.samples.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'class': cls,
                    'original_class': cls  # Сохраняем оригинальный класс
                })
            
            # Визуализация первого примера (только для первого изображения)
            if visualize_first and not visualized and bboxes:
                patches, labels, patch_info = extract_circumscribed_squares(
                    img_path, [bboxes[0]], [classes[0]], target_size, context
                )
                if patch_info:
                    visualize_extraction_process(img_path, bboxes[0], patch_info[0])
                    visualized = True
        
        print(f"Total patches in dataset: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Создаем патч на лету
        patches, labels, patch_info = extract_circumscribed_squares(
            sample['img_path'], 
            [sample['bbox']], 
            [sample['class']], 
            self.target_size, 
            self.context
        )
        
        if not patches:
            # Если патч не удалось извлечь, возвращаем черное изображение
            patch = Image.new('RGB', self.target_size, (0, 0, 0))
            label = -1
        else:
            patch = patches[0]
            label = labels[0]  # Здесь уже будет class-1 (вычитание в extract_circumscribed_squares)
        
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label
    

patch_transform = T.Compose([
    T.Resize(TARGET_SIZE),  # Ресайз до нужного размера
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Стандартные значения для ImageNet
])


def load_dinov2(model_size='small'):
    """Загрузка DINOv2 модели"""
    print(f"Loading DINOv2 {model_size}...")
    
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    feat_dim = 384

    model.to(DEVICE)
    model.eval()
    print(f"DINOv2 loaded. Feature dimension: {feat_dim}")
    
    return model, feat_dim

def extract_features(dataset, model, batch_size=32, device=None):
    """Извлечение фичей из патчей с очисткой памяти"""
    if device is None:
        device = next(model.parameters()).device
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=0)  # Установите num_workers=0 для избежания проблем с памятью
    all_features = []
    all_labels = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch, lbl in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            feats = model(batch)
            
            # Немедленно переносим на CPU и освобождаем GPU память
            all_features.append(feats.cpu().numpy())
            all_labels.append(lbl.numpy())
            
            # Явное удаление тензоров для освобождения памяти
            del batch, feats
            
    
    # Конкатенируем все батчи
    if all_features:
        return np.vstack(all_features), np.concatenate(all_labels)
    else:
        return np.array([]), np.array([])
    


def visualize_features(features, labels, save_path="pca.png"):
    """Визуализация фичей через PCA"""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Берем максимум 500 точек
    if len(features) > 500:
        idx = np.random.choice(len(features), 500, False)
        features = features[idx]
        labels = labels[idx]
    
    # PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                c=labels, cmap='tab10', alpha=0.7, s=50)
    plt.colorbar(label='Class')
    plt.title(f"PCA of Features\nPoints: {len(features)}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    return pca


dataset = PatchDataset(
        annotations=annotations,
        transform=patch_transform,
        max_patches=MAX_PATCHES_PER_IMG,
        context=0.2
    )


model, feat_dim = load_dinov2('small')

# 5. Извлечение фичей
features, labels = extract_features(dataset, model, batch_size=16)

# 6. Визуализация фичей
pca = visualize_features(features, labels, "features.png")


# 7. Сохранение результатов
np.save(os.path.join(SAVE_DIR, "features_circumscribed.npy"), features)
np.save(os.path.join(SAVE_DIR, "labels_circumscribed.npy"), labels)

if labels is not None and len(labels) > 0:
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # Конвертируем numpy int64 в стандартные Python int и ключи в строки
    class_distribution = {str(int(label)): int(count) for label, count in zip(unique_labels, label_counts)}
else:
    class_distribution = {}

class_info = {
    'num_classes': int(len(set(labels)) if labels is not None else 0),
    'class_distribution': class_distribution,  # Теперь ключи - строки
    'feature_dim': int(feat_dim),
    'num_patches': int(len(features) if features is not None else 0),
    'image_size': f"{IMG_WIDTH}x{IMG_HEIGHT}",
    'target_size': f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
    'context_ratio': 0.2,
    'extraction_method': 'circumscribed_squares',
    'dataset_info': {
        'total_images': int(len(annotations)),
        'max_patches_per_image': int(MAX_PATCHES_PER_IMG)
    }
}


# Сохраняем JSON
with open(os.path.join(SAVE_DIR, "class_info_circumscribed.json"), 'w') as f:
    json.dump(class_info, f, indent=2, ensure_ascii=False)