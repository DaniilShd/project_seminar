import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Пути из конфигурации
CSV_PATH = Path(cfg["paths"]["csv_path"])
IMG_DIR = Path(cfg["paths"]["img_dir"])
SAVE_PATH = Path(cfg["paths"]["save_path"])
PATH_VISUALIZE = Path(cfg["paths"]["path_visualize"])

# Параметры изображений
IMG_WIDTH = cfg['image_params']['img_width']
IMG_HEIGHT = cfg['image_params']['img_height']

MIN_CONTOUR_AREA = cfg['extract_patches']['min_contour_area']
MIN_BBOX_SIZE = cfg['extract_patches']['min_bbox_size']
CLASS_IDS = cfg['extract_patches']['class_ids']


def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Преобразует RLE-строку в бинарную маску."""
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, str(mask_rle).split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.asarray(starts) - 1
    lengths = np.asarray(lengths)

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    return mask.reshape((shape[1], shape[0])).T


def build_combined_mask(
    class_rows: pd.DataFrame,
    height: int,
    width: int
) -> np.ndarray:
    """Создает объединенную маску для одного класса."""
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for _, row in class_rows.iterrows():
        mask = rle_decode(row["EncodedPixels"], (height, width))
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask.astype(np.uint8)


def extract_bboxes_from_mask(mask: np.ndarray) -> List[List[int]]:
    """Извлекает bounding boxes из бинарной маски."""
    bboxes: List[List[int]] = []

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(IMG_WIDTH, x + w)
        y_max = min(IMG_HEIGHT, y + h)

        if (x_max - x_min) >= MIN_BBOX_SIZE and (y_max - y_min) >= MIN_BBOX_SIZE:
            bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes


def create_json_annotations(
    csv_path: Path,
    img_dir: Path,
    save_path: Path
) -> Dict[str, Dict[str, List]]:
    """Создает JSON-файл с аннотациями bbox по RLE-разметке."""
    df = pd.read_csv(csv_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    annotations: Dict[str, Dict[str, List]] = {}
    invalid_count = 0

    print(f"Обработка {df['ImageId'].nunique()} изображений...")

    for img_id, group in df.groupby("ImageId"):
        img_path = img_dir / img_id

        if not img_path.exists():
            print(f"Файл не найден: {img_path}")
            invalid_count += 1
            continue

        bboxes: List[List[int]] = []
        classes: List[int] = []

        for class_id in CLASS_IDS:
            class_rows = group[group["ClassId"] == class_id]
            if class_rows.empty:
                continue

            mask = build_combined_mask(class_rows, IMG_HEIGHT, IMG_WIDTH)
            if not mask.any():
                continue

            class_bboxes = extract_bboxes_from_mask(mask)
            bboxes.extend(class_bboxes)
            classes.extend([class_id] * len(class_bboxes))

        if bboxes:
            annotations[img_id] = {
                "image_path": str(img_path),
                "bboxes": bboxes,
                "classes": classes,
            }

    with open(save_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Успешно обработано: {len(annotations)}")
    print(f"Пропущено файлов: {invalid_count}")

    return annotations


def visualize_annotations(
    annotations: Dict[str, Dict],
    output_dir: Path,
    num_samples: int = 3
) -> None:
    """Сохраняет изображения с нарисованными bounding boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {1: "red", 2: "green", 3: "blue", 4: "yellow"}

    for img_id in list(annotations.keys())[:num_samples]:
        ann = annotations[img_id]
        img = cv2.imread(ann["image_path"])

        if img is None:
            print(f"Не удалось загрузить {ann['image_path']}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb)

        for bbox, class_id in zip(ann["bboxes"], ann["classes"]):
            x_min, y_min, x_max, y_max = bbox
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=colors.get(class_id, "white"),
                facecolor="none",
            )
            ax.add_patch(rect)
            
            ax.text(x_min - 10, y_min - 5, str(class_id), 
                    color=colors.get(class_id, "white"),
                    fontsize=8, fontweight='bold')

        ax.set_title(f"{img_id} — {len(ann['bboxes'])} объектов")
        ax.axis("off")

        save_path = output_dir / f"{img_id}_annotated.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    annotations = create_json_annotations(CSV_PATH, IMG_DIR, SAVE_PATH)

    if annotations:
        visualize_annotations(annotations, PATH_VISUALIZE)

        stats_path = SAVE_PATH.parent / "annotations_stats.txt"
        total_boxes = sum(len(a["bboxes"]) for a in annotations.values())

        with open(stats_path, "w") as f:
            f.write(f"Изображений: {len(annotations)}\n")
            f.write(f"Всего bbox: {total_boxes}\n")

        print(f"Статистика сохранена: {stats_path}")
