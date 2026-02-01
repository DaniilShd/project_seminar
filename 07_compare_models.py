import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import time

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = (cfg["training"]["models_dir"])
SAVE_DIR = Path(cfg["paths"]["save_dir"])
REPORT_DIR = Path(cfg["paths"]["report_dir"])
PATH_VISUALIZE = Path(cfg["paths"]["path_visualize"])

os.makedirs(PATH_VISUALIZE, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("Comparing models")

features = np.load(f"{SAVE_DIR}/features.npy")
labels = np.load(f"{SAVE_DIR}/labels.npy")

num_classes = len(np.unique(labels))
input_dim = features.shape[1]

X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

class SimpleNN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,num_classes)
        )
    def forward(self,x): return self.net(x)

class TinyStudent(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,num_classes))
    def forward(self,x): return self.net(x)

results = []
table_data = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def format_time(seconds):
    if seconds < 1e-6:  # наносекунды
        return f"{seconds*1e9:.2f}ns"
    elif seconds < 1e-3:  # микросекунды
        return f"{seconds*1e6:.2f}µs"
    elif seconds < 1:  # миллисекунды
        return f"{seconds*1e3:.2f}ms"
    else:  # секунды
        return f"{seconds:.4f}s"

# Linear Probe Original
start_time = time.time()
lr = joblib.load(os.path.join(MODELS_DIR,"linear_probe.pkl"))
predict_time = time.time()
y_pred_lr = lr.predict(X_test)
inference_time = time.time() - predict_time
acc_lr = accuracy_score(y_test, y_pred_lr) * 100
params_lr = X_test.shape[1] * num_classes + num_classes 
results.append(("Linear Probe (Original)", acc_lr))
table_data.append(["Linear Probe (Original)", params_lr, f"{acc_lr:.2f}%", format_time(inference_time), "NaN", "NaN"])

# Teacher
teacher = SimpleNN(input_dim, num_classes).to(DEVICE)
teacher.load_state_dict(torch.load(os.path.join(MODELS_DIR,"teacher_nn.pth")))
teacher.eval()
start_time = time.time()
with torch.no_grad():
    preds_teacher = teacher(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
inference_time = time.time() - start_time
acc_teacher = accuracy_score(y_test, preds_teacher) * 100
params_teacher = count_parameters(teacher)
results.append(("Teacher NN", acc_teacher))
table_data.append(["Neural Network (Teacher)", params_teacher, f"{acc_teacher:.2f}%", format_time(inference_time), "NaN", "NaN"])

# Student
student = TinyStudent(input_dim, num_classes).to(DEVICE)
student.load_state_dict(torch.load(os.path.join(MODELS_DIR,"student_nn.pth")))
student.eval()
start_time = time.time()
with torch.no_grad():
    preds_student = student(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
inference_time = time.time() - start_time
acc_student = accuracy_score(y_test, preds_student) * 100
params_student = count_parameters(student)
compression_ratio = params_teacher / params_student if params_teacher > 0 else "NaN"
improvement = (acc_student - acc_teacher) if params_teacher > 0 else "NaN"
results.append(("Student NN", acc_student))
improvement_str = f"{improvement:+.2f}%" if improvement != "NaN" else "NaN"
table_data.append(["Tiny Student (Distilled)", params_student, f"{acc_student:.2f}%", format_time(inference_time), f"{compression_ratio:.1f}x", improvement_str])

# SSL Linear Probe
if os.path.exists(f"{SAVE_DIR}/features_ssl.npy"):
    features_ssl = np.load(f"{SAVE_DIR}/features_ssl.npy")
    labels_ssl = np.load(f"{SAVE_DIR}/labels_ssl.npy")
    _, X_test_ssl, _, y_test_ssl = train_test_split(
        features_ssl, labels_ssl, test_size=0.2, random_state=42, stratify=labels_ssl
    )
    start_time = time.time()
    lr_ssl = joblib.load(os.path.join(MODELS_DIR,"linear_probe_ssl.pkl"))
    predict_time = time.time()
    y_pred_ssl = lr_ssl.predict(X_test_ssl)
    inference_time = time.time() - predict_time
    acc_ssl = accuracy_score(y_test_ssl, y_pred_ssl) * 100
    params_ssl = X_test_ssl.shape[1] * num_classes + num_classes
    results.append(("Linear Probe (SSL)", acc_ssl))
    improvement = acc_ssl - acc_lr
    improvement_str = f"{improvement:+.2f}%" if improvement != "NaN" else "NaN"
    table_data.append(["Linear Probe (SSL Finetuned)", params_ssl, f"{acc_ssl:.2f}%", format_time(inference_time), "NaN", improvement_str])

# Вывод таблицы
print("RESULTS TABLE:")
print(f"{'Model':<35} {'Parameters':<12} {'Accuracy':<10} {'Inference Time':<15} {'Compression':<12} {'Improvement':<10}")
for row in table_data:
    print(f"{row[0]:<35} {row[1]:<12} {row[2]:<10} {row[3]:<15} {row[4]:<12} {row[5]:<10}")

# Сохранение в текстовом формате
with open(os.path.join(REPORT_DIR, "results_table.txt"), "w") as f:
    f.write("📋 RESULTS TABLE:\n")
    f.write("=" * 90 + "\n")
    f.write(f"{'Model':<35} {'Parameters':<12} {'Accuracy':<10} {'Inference Time':<15} {'Compression':<12} {'Improvement':<10}\n")
    f.write("=" * 90 + "\n")
    for row in table_data:
        f.write(f"{row[0]:<35} {row[1]:<12} {row[2]:<10} {row[3]:<15} {row[4]:<12} {row[5]:<10}\n")
    f.write("=" * 90 + "\n")

# Сохранение в csv
df = pd.DataFrame(results, columns=["Model","Test Accuracy"])
df.to_csv(os.path.join(REPORT_DIR,"results.csv"), index=False)

# Словарь для хранения всех матриц
confusion_matrices = {}

# 1. Confusion Matrix для Linear Probe Original
cm_lr = confusion_matrix(y_test, y_pred_lr)
confusion_matrices["Linear Probe (Original)"] = cm_lr

plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Linear Probe (Original)\nAccuracy: {acc_lr:.2f}%")
plt.tight_layout()
plt.savefig(os.path.join(PATH_VISUALIZE, "confusion_matrix_lr.png"))
plt.close()

# 2. Confusion Matrix для Teacher
cm_teacher = confusion_matrix(y_test, preds_teacher)
confusion_matrices["Teacher NN"] = cm_teacher

plt.figure(figsize=(8,6))
sns.heatmap(cm_teacher, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Neural Network (Teacher)\nAccuracy: {acc_teacher:.2f}%")
plt.tight_layout()
plt.savefig(os.path.join(PATH_VISUALIZE, "confusion_matrix_teacher.png"))
plt.close()

# 3. Confusion Matrix для Student
cm_student = confusion_matrix(y_test, preds_student)
confusion_matrices["Student NN"] = cm_student

plt.figure(figsize=(8,6))
sns.heatmap(cm_student, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - Tiny Student (Distilled)\nAccuracy: {acc_student:.2f}%")
plt.tight_layout()
plt.savefig(os.path.join(PATH_VISUALIZE, "confusion_matrix_student.png"))
plt.close()

# 4. Confusion Matrix для SSL 
if os.path.exists(f"{SAVE_DIR}/features_ssl.npy"):
    cm_ssl = confusion_matrix(y_test_ssl, y_pred_ssl)
    confusion_matrices["Linear Probe (SSL)"] = cm_ssl
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_ssl, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Linear Probe (SSL Finetuned)\nAccuracy: {acc_ssl:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_VISUALIZE, "confusion_matrix_ssl.png"))
    plt.close()

# Сохранение всех матриц в файл numpy для дальнейшего анализа
np.savez(os.path.join(REPORT_DIR, "confusion_matrices.npz"), **confusion_matrices)

# Сохранение матриц в текстовом формате
with open(os.path.join(REPORT_DIR, "confusion_matrices.txt"), "w") as f:
    for model_name, cm in confusion_matrices.items():
        f.write(f"\n{'='*60}\n")
        f.write(f"Confusion Matrix: {model_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(str(cm))
        f.write("\n\n")



# Создание сводной визуализации с 4 subplots (если есть все 4 модели)
if len(confusion_matrices) >= 3:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, y=1.02)
    
    models_to_plot = []
    accuracies_to_plot = []
    cms_to_plot = []
    
    if "Linear Probe (Original)" in confusion_matrices:
        models_to_plot.append("Linear Probe")
        accuracies_to_plot.append(acc_lr)
        cms_to_plot.append(cm_lr)
    
    if "Teacher NN" in confusion_matrices:
        models_to_plot.append("Teacher NN")
        accuracies_to_plot.append(acc_teacher)
        cms_to_plot.append(cm_teacher)
    
    if "Student NN" in confusion_matrices:
        models_to_plot.append("Student NN")
        accuracies_to_plot.append(acc_student)
        cms_to_plot.append(cm_student)
    
    if "Linear Probe (SSL)" in confusion_matrices:
        models_to_plot.append("SSL Finetuned")
        accuracies_to_plot.append(acc_ssl)
        cms_to_plot.append(cm_ssl)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(models_to_plot):
            sns.heatmap(cms_to_plot[idx], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{models_to_plot[idx]}\nAccuracy: {accuracies_to_plot[idx]:.2f}%")
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_VISUALIZE, "all_confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created combined confusion matrices visualization")
