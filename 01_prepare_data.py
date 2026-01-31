import pandas as pd
from pathlib import Path
import numpy as np
import json
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Пути из конфигурации
CSV_PATH = cfg['paths']['csv_path']
IMG_DIR = cfg['paths']['img_dir']
SAVE_PATH = cfg['paths']['save_path']
PATH_VISUALIZE = cfg['paths']['path_visualize']

# Параметры изображений
IMG_WIDTH = cfg['image_params']['img_width']
IMG_HEIGHT = cfg['image_params']['img_height']


def rle_decode(mask_rle, shape=(256, 1600)):
    """Декодирует RLE строку в бинарную маску"""
    if isinstance(mask_rle, float) and np.isnan(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    
    if pd.isna(mask_rle):  # Дополнительная проверка
        return np.zeros(shape, dtype=np.uint8)
    
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    
    return mask.reshape(shape[::-1]).T

def create_json_annotations(csv_path, img_dir, save_path):
    """
    Создает JSON аннотации в формате:
    annotations[img_id] = {
        'image_path': img_path,
        'bboxes': bboxes,
        'classes': classes
    }
    """
    # Читаем CSV
    df = pd.read_csv(csv_path)
    img_dir_path = Path(img_dir)
    
    # Создаем папку для сохранения
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    annotations = {}
    invalid_count = 0
    total_images = df['ImageId'].nunique()
    
    print(f"Обработка {total_images} изображений...")
    
    # Группируем по изображениям
    for img_id, group in df.groupby('ImageId'):
        img_path = str(img_dir_path / img_id)
        
        # Проверяем существование файла
        if not Path(img_path).exists():
            print(f"Предупреждение: файл не найден {img_path}")
            invalid_count += 1
            continue
        
        bboxes = []
        classes = []
        
        # Группируем по классам
        for class_id in [1, 2, 3, 4]:
            class_rows = group[group['ClassId'] == class_id]
            
            if len(class_rows) == 0:
                continue
            
            # Создаем комбинированную маску для этого класса
            combined_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            
            for _, row in class_rows.iterrows():
                rle = row['EncodedPixels']
                mask = rle_decode(rle, shape=(IMG_HEIGHT, IMG_WIDTH))
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            
            # Если маска пустая, пропускаем
            if np.sum(combined_mask) == 0:
                continue
            
            # Находим контуры в маске
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Обрабатываем каждый контур отдельно
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Фильтр по минимальной площади (опционально)
                if area < 32:  # минимальная площадь 32 пикселя
                    continue
                
                # Получаем bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Проверка валидности координат
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(IMG_WIDTH, x + w)
                y_max = min(IMG_HEIGHT, y + h)
                
                # Проверка минимального размера bbox
                if (x_max - x_min) >= 8 and (y_max - y_min) >= 8:
                    bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                    classes.append(int(class_id))
        
        # Сохраняем только если есть bbox
        if bboxes:
            annotations[img_id] = {
                'image_path': img_path,
                'bboxes': bboxes,
                'classes': classes
            }
    
    # Сохраняем в JSON
    with open(save_path, 'w') as f:
        json.dump(annotations, f, indent=2, separators=(',', ': '))
    
    # Статистика
    print(f"Всего изображений в CSV: {total_images}")
    print(f"Успешно обработано: {len(annotations)}")
    print(f"Пропущено: {invalid_count}")
    
    if annotations:
        total_bboxes = sum(len(a['bboxes']) for a in annotations.values())
        print(f"Всего bbox: {total_bboxes}")
        
        # Распределение по классам
        all_classes = []
        for ann in annotations.values():
            all_classes.extend(ann['classes'])
        
        if all_classes:
            unique_classes, counts = np.unique(all_classes, return_counts=True)
            print(f"\nРаспределение по классам:")
            for cls, count in zip(unique_classes, counts):
                print(f"  Класс {cls}: {count} bbox ({count/len(all_classes)*100:.1f}%)")
        
        # Среднее количество bbox на изображение
        avg_bbox_per_image = total_bboxes / len(annotations)
        print(f"Среднее bbox на изображение: {avg_bbox_per_image:.2f}")
        
        # Сохраняем примеры для проверки
        print(f"\nПримеры первых 3 изображений:")
        for i, (img_id, ann) in enumerate(list(annotations.items())[:3]):
            print(f"  {img_id}: {len(ann['bboxes'])} bbox, классы: {set(ann['classes'])}")
    
    print(f"\nJSON сохранен: {save_path}")
    return annotations


def visualize_annotations(annotations, path_visualize, num_samples=3):
    """Сохранение нескольких примеров с аннотациями в указанную папку"""
    
    # Создаем папку, если она не существует
    os.makedirs(path_visualize, exist_ok=True)
    
    sample_keys = list(annotations.keys())[:num_samples]
    
    for img_id in sample_keys:
        ann = annotations[img_id]
        img_path = ann['image_path']
        
        # Загружаем изображение
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось загрузить: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Исходное изображение
        ax1.imshow(img_rgb)
        ax1.set_title(f'Исходное: {img_id}')
        ax1.axis('off')
        
        # С bbox
        ax2.imshow(img_rgb)
        
        # Цвета для классов
        colors = {
            1: 'red',
            2: 'green', 
            3: 'blue',
            4: 'yellow'
        }
        
        class_names = {
            1: 'Class 1',
            2: 'Class 2',
            3: 'Class 3',
            4: 'Class 4'
        }
        
        # Рисуем bbox
        for bbox, class_id in zip(ann['bboxes'], ann['classes']):
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            
            color = colors.get(class_id, 'white')
            
            # Прямоугольник
            rect = Rectangle((x_min, y_min), width, height,
                            linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Подпись
            label = class_names.get(class_id, f'Class {class_id}')
            ax2.text(x_min, max(0, y_min-5), label,
                    color=color, fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax2.set_title(f'С bbox: {len(ann["bboxes"])} объектов')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Сохраняем изображение вместо отображения
        save_path = os.path.join(path_visualize, f"{img_id}_annotated.png")
        plt.savefig(save_path)
        plt.close(fig)  




if __name__ == "__main__":
    # Создаем JSON аннотации
    annotations = create_json_annotations(
        csv_path=CSV_PATH,
        img_dir=IMG_DIR,
        save_path=SAVE_PATH
    )

    # Визуализируем примеры
    if annotations:
        visualize_annotations(annotations, PATH_VISUALIZE, num_samples=3)
        
        # Дополнительно: создаем простую текстовую статистику
        stats_path = Path(SAVE_PATH).parent / "annotations_stats.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Всего изображений: {len(annotations)}\n")
            f.write(f"Всего bbox: {sum(len(a['bboxes']) for a in annotations.values())}\n\n")
            
            # Распределение по классам
            all_classes = []
            for ann in annotations.values():
                all_classes.extend(ann['classes'])
            
            if all_classes:
                unique_classes, counts = np.unique(all_classes, return_counts=True)
                f.write("Распределение по классам:\n")
                for cls, count in zip(unique_classes, counts):
                    f.write(f"  Класс {cls}: {count}\n")
        
        print(f"\nСтатистика сохранена: {stats_path}")
