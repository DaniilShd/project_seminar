import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PATCH_DIR = Path(cfg["extract_patches"]["save_dir_patches"])
ANNOTATION_FILE = Path(cfg["extract_patches"]["annotation_path"])
SAVE_DIR = Path(cfg["paths"]["save_dir"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TARGET_SIZE = cfg["ssl_dino_finetune"]["target_size"]
BATCH_SIZE = cfg["ssl_dino_finetune"]["batch_size"]
EPOCHS = cfg["ssl_dino_finetune"]["epochs"]
LEARNING_RATE = float(cfg["ssl_dino_finetune"]["learning_rate"])
WEIGHT_DECAY = float(cfg["ssl_dino_finetune"]["weight_decay"])
FREEZE_BLOCKS = int(cfg["ssl_dino_finetune"]["freeze_blocks"])

class SSLTransform:
    def __init__(self, size=TARGET_SIZE):
        self.global_transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.32, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(23),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.local_transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.05, 0.32)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __call__(self, img):
        return self.global_transform(img), self.local_transform(img)

class SSLPatchDataset(Dataset):
    def __init__(self, ann_path, patches_dir):
        with open(ann_path) as f:
            data = json.load(f)

        self.names = list(data.keys())
        self.dir = Path(patches_dir)
        self.transform = SSLTransform()

        print(f"SSL patches found: {len(self.names)}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(self.dir / self.names[idx]).convert("RGB")
        return self.transform(img)

class DINOHead(nn.Module):
    def __init__(self, in_dim=384, out_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class DINOWithMomentum(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.student_head = DINOHead()
        self.teacher_head = DINOHead()

        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.register_buffer("center", torch.zeros(384))
        self.momentum = 0.996
        self.center_momentum = 0.9
        self.student_temp = 0.1
        self.teacher_temp = 0.04

    def update_teacher(self):
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data.mul_(self.momentum).add_(ps.data, alpha=1 - self.momentum)
            for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                pt.data.mul_(self.momentum).add_(ps.data, alpha=1 - self.momentum)

    def forward(self, x1, x2):
        with torch.no_grad():
            t1_logits = self.teacher_head(self.teacher(x1))
            t2_logits = self.teacher_head(self.teacher(x2))
            t1 = torch.softmax((t1_logits - self.center) / self.teacher_temp, dim=-1)
            t2 = torch.softmax((t2_logits - self.center) / self.teacher_temp, dim=-1)

        s1 = self.student_head(self.student(x1))
        s2 = self.student_head(self.student(x2))
        return (s1, s2), (t1, t2), (t1_logits, t2_logits)

    def update_center(self, teacher_logits):
        with torch.no_grad():
            batch_center = torch.cat(teacher_logits).mean(dim=0)
            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

def train_ssl(model, loader, epochs):
    for name, param in model.student.named_parameters():
        if any(f"blocks.{i}" in name for i in range(0, FREEZE_BLOCKS)):  
            param.requires_grad = False
        else:
            param.requires_grad = True

        if "patch_embed" in name:
            param.requires_grad = False

    print("Trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    params = [p for p in model.student.parameters() if p.requires_grad] + \
             list(model.student_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for v1, v2 in tqdm(loader, desc=f"SSL Epoch {epoch+1}"):
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            (s1, s2), (t1, t2), teacher_logits = model(v1, v2)

            s1_log = torch.log_softmax(s1 / model.student_temp, dim=-1)
            s2_log = torch.log_softmax(s2 / model.student_temp, dim=-1)

            loss = (nn.functional.kl_div(s1_log, t2.detach(), reduction='batchmean') +
                    nn.functional.kl_div(s2_log, t1.detach(), reduction='batchmean')) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_teacher()
            model.update_center(teacher_logits)
            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

    return model

if __name__ == "__main__":
    dataset = SSLPatchDataset(ANNOTATION_FILE, PATCH_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = DINOWithMomentum().to(DEVICE)
    model = train_ssl(model, loader, epochs=EPOCHS)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ssl_model_path = SAVE_DIR / "dinov2_ssl_finetuned.pth"

    torch.save({'student_state_dict': model.student.state_dict()}, ssl_model_path)
    print(f"SSL model saved to {ssl_model_path}")
