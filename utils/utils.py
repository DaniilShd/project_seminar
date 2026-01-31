import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

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