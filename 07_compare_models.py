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

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = (cfg["training"]["models_dir"])
SAVE_DIR = Path(cfg["paths"]["save_dir"])
REPORT_DIR = Path(cfg["paths"]["report_dir"])
PATH_VISUALIZE = Path(cfg["paths"]["path_visualize"])

os.makedirs(PATH_VISUALIZE, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("📊 Comparing models on TEST set...")

# ================= LOAD ORIGINAL FEATURES =================
features = np.load(f"{SAVE_DIR}/features.npy")
labels = np.load(f"{SAVE_DIR}/labels.npy")

num_classes = len(np.unique(labels))
input_dim = features.shape[1]

X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# ================= MODELS =================
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

# 1️⃣ Linear Probe Original
lr = joblib.load(os.path.join(MODELS_DIR,"linear_probe.pkl"))
acc_lr = accuracy_score(y_test, lr.predict(X_test))
results.append(("Linear Probe (Original)", acc_lr))

# 2️⃣ Teacher
teacher = SimpleNN(input_dim, num_classes).to(DEVICE)
teacher.load_state_dict(torch.load(os.path.join(MODELS_DIR,"teacher_nn.pth")))
teacher.eval()
with torch.no_grad():
    preds_teacher = teacher(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
results.append(("Teacher NN", accuracy_score(y_test, preds_teacher)))

# 3️⃣ Student
student = TinyStudent(input_dim, num_classes).to(DEVICE)
student.load_state_dict(torch.load(os.path.join(MODELS_DIR,"student_nn.pth")))
student.eval()
with torch.no_grad():
    preds_student = student(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
results.append(("Student NN", accuracy_score(y_test, preds_student)))

# 4️⃣ SSL Linear Probe
if os.path.exists(f"{SAVE_DIR}/features_ssl.npy"):
    features_ssl = np.load(f"{SAVE_DIR}/features_ssl.npy")
    labels_ssl = np.load(f"{SAVE_DIR}/labels_ssl.npy")
    _, X_test_ssl, _, y_test_ssl = train_test_split(
        features_ssl, labels_ssl, test_size=0.2, random_state=42, stratify=labels_ssl
    )
    lr_ssl = joblib.load(os.path.join(MODELS_DIR,"linear_probe_ssl.pkl"))
    acc_ssl = accuracy_score(y_test_ssl, lr_ssl.predict(X_test_ssl))
    results.append(("Linear Probe (SSL)", acc_ssl))

# ================= SAVE REPORT =================
df = pd.DataFrame(results, columns=["Model","Test Accuracy"])
df.to_csv(os.path.join(REPORT_DIR,"results.csv"), index=False)
print(df)

# ================= BAR CHART =================
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Test Accuracy"])
plt.ylim(0,1)
plt.title("Model Comparison (TEST)")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PATH_VISUALIZE,"accuracy_comparison.png"))
plt.close()

# ================= CONFUSION MATRIX (BEST MODEL) =================
best_name, _ = max(results, key=lambda x: x[1])
print(f"🏆 Best model: {best_name}")

if "Original" in best_name:
    y_pred = lr.predict(X_test)
elif "Teacher" in best_name:
    y_pred = preds_teacher
elif "Student" in best_name:
    y_pred = preds_student
else:
    y_pred = lr_ssl.predict(X_test_ssl)
    y_test = y_test_ssl

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix ({best_name})")
plt.tight_layout()
plt.savefig(os.path.join(PATH_VISUALIZE,"confusion_matrix.png"))
plt.close()

print("\n✅ Comparison complete. Reports in ./report, plots in ./visualize")
