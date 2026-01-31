import subprocess
import zipfile
import os
import sys

# 1. Создаём папку 'dataset'
os.makedirs("dataset", exist_ok=True)

# 2. Скачиваем датасет
subprocess.run(["kaggle", "competitions", "download", "-c", "severstal-steel-defect-detection"])

# 3. Распаковываем в папку 'dataset'
with zipfile.ZipFile("severstal-steel-defect-detection.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

# 4. Удаляем архив
os.remove("severstal-steel-defect-detection.zip")