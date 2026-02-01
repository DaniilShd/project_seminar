import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
import joblib
from pathlib import Path
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = (cfg["training"]["models_dir"])
SAVE_DIR = Path(cfg["paths"]["save_dir"])

os.makedirs(MODELS_DIR, exist_ok=True)

print("Training models...")

# ================= LOAD FEATURES =================
features = np.load(os.path.join(SAVE_DIR, "features.npy"))
labels = np.load(os.path.join(SAVE_DIR, "labels.npy"))

SSL_AVAILABLE = False
if os.path.exists(os.path.join(SAVE_DIR, "features_ssl.npy")):
    features_ssl = np.load(os.path.join(SAVE_DIR, "features_ssl.npy"))
    labels_ssl = np.load(os.path.join(SAVE_DIR, "labels_ssl.npy"))
    SSL_AVAILABLE = True

num_classes = len(np.unique(labels))
input_dim = features.shape[1]


X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=32, shuffle=True
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
            nn.Linear(32,num_classes)
        )
    def forward(self,x): return self.net(x)

def train_model(model, epochs=15):
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, loss {total/len(train_loader):.4f}")
    return model

def distill(student, teacher, epochs=20):
    opt = optim.Adam(student.parameters(), lr=0.01)
    T = 3.0
    alpha = 0.7
    teacher.eval()
    for epoch in range(epochs):
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student(xb)
            loss = alpha * F.kl_div(
                F.log_softmax(s_logits/T, dim=1),
                F.softmax(t_logits/T, dim=1),
                reduction='batchmean') * (T*T) \
                   + (1-alpha) * F.cross_entropy(s_logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Distill Epoch {epoch+1}, loss {total/len(train_loader):.4f}")
    return student

# LINEAR PROBE (ORIGINAL)
print("\nLinear Probe (Original)")
lr = LogisticRegression(max_iter=1000, n_jobs=-1)
lr.fit(X_train, y_train)
joblib.dump(lr, os.path.join(MODELS_DIR, "linear_probe.pkl"))

# TEACHER 
print("\nTeacher NN")
teacher = train_model(SimpleNN(input_dim, num_classes).to(DEVICE))
torch.save(teacher.state_dict(), os.path.join(MODELS_DIR, "teacher_nn.pth"))

# STUDENT 
print("\nDistilled Student")
student = distill(TinyStudent(input_dim, num_classes).to(DEVICE), teacher)
torch.save(student.state_dict(), os.path.join(MODELS_DIR, "student_nn.pth"))

# SSL LINEAR PROBE 
if SSL_AVAILABLE:
    print("\n🔹 Linear Probe (SSL features)")
    X_train_ssl, X_val_ssl, y_train_ssl, y_val_ssl = train_test_split(
        features_ssl, labels_ssl, test_size=0.3, random_state=42, stratify=labels_ssl
    )
    lr_ssl = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr_ssl.fit(X_train_ssl, y_train_ssl)
    joblib.dump(lr_ssl, os.path.join(MODELS_DIR, "linear_probe_ssl.pkl"))

print("\nAll models trained and saved in ./models")