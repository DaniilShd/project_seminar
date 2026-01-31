import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from tqdm import tqdm  # <- добавили tqdm


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ANNOTATION_PATH = Path(cfg["paths"]["annotation_path"])
ANNOTATION_PATH_PATCHES = Path(cfg["extract_patches"]["annotation_path"])
SAVE_DIR = Path(cfg["extract_patches"]["save_dir_patches"])
PATH_VISUALIZE = Path(cfg["paths"]["path_visualize"])
TARGET_SIZE = (cfg["image_params"]["target_width"], cfg["image_params"]["target_height"])
CONTEXT = cfg["extract_patches"].get("context", 0.2)
MAX_PATCHES_PER_IMG = cfg["extract_patches"].get("max_patches_per_img", 5)



def extract_circumscribed_square(image_path, bbox, target_size=TARGET_SIZE, context=CONTEXT):
    """Вырезает квадратный патч вокруг bbox с контекстом."""
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    x1, y1, x2, y2 = bbox

    defect_w = x2 - x1
    defect_h = y2 - y1

    square_side = max(defect_w, defect_h)
    square_side = int(square_side * (1 + 2 * context))
    square_side = min(square_side, img_w, img_h)
    square_side = max(square_side, 16)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    sx1 = max(0, center_x - square_side // 2)
    sy1 = max(0, center_y - square_side // 2)
    sx2 = min(img_w, sx1 + square_side)
    sy2 = min(img_h, sy1 + square_side)

    patch = img.crop((sx1, sy1, sx2, sy2))
    patch = patch.resize(target_size)

    return patch, (sx1, sy1, sx2, sy2)


def save_patch(patch, img_id, idx, save_dir):
    """Сохраняет патч в указанную папку и возвращает путь."""
    save_dir.mkdir(parents=True, exist_ok=True)
    patch_path = save_dir / f"{img_id}_patch_{idx}.png"
    patch.save(patch_path)
    return str(patch_path)


def visualize_patch_extraction(img_path, bbox, square_bbox, patch, save_dir):
    """Визуализирует процесс вырезания патча."""
    save_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Оригинал с bbox
    axes[0].imshow(img)
    x1, y1, x2, y2 = bbox
    axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=2))
    axes[0].set_title("Original bbox")

    # 2. Квадрат с контекстом
    axes[1].imshow(img)
    sx1, sy1, sx2, sy2 = square_bbox
    axes[1].add_patch(plt.Rectangle((sx1, sy1), sx2-sx1, sy2-sy1, fill=False, edgecolor="blue", linewidth=2))
    axes[1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=2))
    axes[1].set_title("Circumscribed square")

    # 3. Финальный патч
    axes[2].imshow(patch)
    axes[2].set_title("Extracted patch")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    filename = Path(img_path).stem
    save_path = save_dir / f"{filename}_extraction.png"
    plt.savefig(save_path)
    plt.close(fig)



if __name__ == "__main__":
    with open(ANNOTATION_PATH, "r") as f:
        data = json.load(f)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PATH_VISUALIZE.mkdir(parents=True, exist_ok=True)

    patches_annotation = {}
    total_patches = 0

    # Прогресс-бар для изображений
    for img_id, info in tqdm(data.items(), desc="Processing images"):
        img_path = Path(info["image_path"])
        bboxes = info["bboxes"][:MAX_PATCHES_PER_IMG]
        classes = info["classes"][:MAX_PATCHES_PER_IMG]

        # Прогресс-бар для патчей внутри изображения
        for idx, (bbox, cls) in enumerate(tqdm(zip(bboxes, classes), total=len(bboxes), desc=f"{img_id} patches", leave=False)):
            patch, square_bbox = extract_circumscribed_square(img_path, bbox)
            patch_path = save_patch(patch, img_id, idx + 1, SAVE_DIR)

            # Сохраняем аннотацию для патча
            patches_annotation[Path(patch_path).name] = {
                "original_image": str(img_path),
                "class": cls,
                "original_bbox": bbox,
                "square_bbox": square_bbox
            }

            total_patches += 1

            # Визуализируем каждый N-й патч
            if total_patches % 3000 == 0:
                visualize_patch_extraction(img_path, bbox, square_bbox, patch, PATH_VISUALIZE)

    # Сохраняем аннотацию к патчам
    annotation_file = ANNOTATION_PATH_PATCHES / "patches_annotation.json"
    with open(annotation_file, "w") as f:
        json.dump(patches_annotation, f, indent=2)

    print(f"Total patches processed: {total_patches}")
    print(f"Saved patches in: {SAVE_DIR}")
    print(f"Visualizations saved in: {PATH_VISUALIZE}")
    print(f"Patches annotation saved in: {annotation_file}")
