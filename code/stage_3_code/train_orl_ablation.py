#!/usr/bin/env python3
"""
train_orl_dropout_full.py

Train ORLNet with dropout and report full metrics on both train & test each epoch.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_full(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            losses.append(criterion(out, y).item())
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc      = accuracy_score(all_labels, all_preds)

    def prf(avg):
        return (
            precision_score(all_labels, all_preds, average=avg, zero_division=0),
            recall_score(all_labels, all_preds, average=avg, zero_division=0),
            f1_score(all_labels, all_preds, average=avg, zero_division=0),
        )

    stats = {'loss': avg_loss, 'accuracy': acc}
    for avg in ('micro', 'macro', 'weighted'):
        p, r, f = prf(avg)
        stats[f'precision_{avg}'] = p
        stats[f'recall_{avg}']    = r
        stats[f'f1_{avg}']        = f
    return stats

class ORLNetDropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.pool    = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        flat = 32 * (112//4) * (92//4)
        self.fc1     = nn.Linear(flat, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def load_orl(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    import torch
    # stack and label‐shift
    tr = np.stack([i['image'] for i in data['train']], axis=0)
    tl = np.array([i['label'] for i in data['train']]) - 1
    te = np.stack([i['image'] for i in data['test']],  axis=0)
    tl2= np.array([i['label'] for i in data['test']])  - 1
    def to_tensor(imgs):
        t = torch.tensor(imgs, dtype=torch.float32)/255.0
        if t.ndim==4 and t.shape[-1]==3: t = t[...,0]
        return t.unsqueeze(1)
    Xtr = to_tensor(tr); ytr = torch.tensor(tl,  dtype=torch.long)
    Xte = to_tensor(te); yte = torch.tensor(tl2, dtype=torch.long)
    return (Xtr,ytr),(Xte,yte)

def main():
    # locate ORL pickle in project/data
    script_dir   = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    data_path    = os.path.join(project_root, 'data', 'ORL')

    # load and wrap
    (Xtr,ytr), (Xte,yte) = load_orl(data_path)
    train_loader = DataLoader(TensorDataset(Xtr,ytr), batch_size=16, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte,yte), batch_size=16)

    # device & model
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nc          = len(np.unique(ytr.numpy()))
    model       = ORLNetDropout(nc).to(device)

    # opt & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # train & evaluate each epoch
    epochs = 50
    for ep in range(1, epochs+1):
        model.train()
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss= criterion(out, y)
            loss.backward()
            optimizer.step()

        # compute stats on both sets
        tr_stats = evaluate_full(model, train_loader, criterion, device)
        te_stats = evaluate_full(model, test_loader,  criterion, device)

        print(f"\nEpoch {ep}/{epochs}")
        print(f" TRAIN — loss: {tr_stats['loss']:.4f}, acc: {tr_stats['accuracy']:.4f}")
        for avg in ('micro','macro','weighted'):
            print(f"    {avg:>8} → P={tr_stats[f'precision_{avg}']:.4f}, "
                  f"R={tr_stats[f'recall_{avg}']:.4f}, F1={tr_stats[f'f1_{avg}']:.4f}")

        print(f" TEST  — loss: {te_stats['loss']:.4f}, acc: {te_stats['accuracy']:.4f}")
        for avg in ('micro','macro','weighted'):
            print(f"    {avg:>8} → P={te_stats[f'precision_{avg}']:.4f}, "
                  f"R={te_stats[f'recall_{avg}']:.4f}, F1={te_stats[f'f1_{avg}']:.4f}")

if __name__ == '__main__':
    main()