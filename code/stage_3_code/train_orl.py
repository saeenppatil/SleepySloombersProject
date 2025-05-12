import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def evaluate_full(model, loader, criterion, device):
    """
    Runs one full pass over `loader`, computing:
      - avg loss
      - accuracy
      - precision (micro, macro, weighted)
      - recall    (micro, macro, weighted)
      - f1-score  (micro, macro, weighted)
    Returns a dict of all these.
    """
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            losses.append(loss.item())

            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # aggregate
    avg_loss = sum(losses) / len(losses)
    acc      = accuracy_score(all_labels, all_preds)

    # helper to build a sub-dict of precision/recall/f1 for one averaging method
    def prf(avg):
        return {
            f"precision_{avg}": precision_score(all_labels, all_preds, average=avg, zero_division=0),
            f"recall_{avg}":    recall_score(all_labels, all_preds, average=avg, zero_division=0),
            f"f1_{avg}":        f1_score(all_labels, all_preds, average=avg, zero_division=0),
        }

    metrics = {
        "loss":     avg_loss,
        "accuracy": acc,
    }
    # merge in micro, macro, weighted
    for avg in ("micro", "macro", "weighted"):
        metrics.update(prf(avg))

    return metrics

# 1. Build the path to your ORL pickle:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
orl_path     = os.path.join(project_root, 'data', 'ORL')   # or 'ORL.pkl' if that's the name

# 2. Load the pickle:
with open(orl_path, 'rb') as f:
    data = pickle.load(f)

# 3. Extract images & labels for train and test:
#    Assume each 'image' is a NumPy array H×W or H×W×C, and 'label' is an int.
import numpy as np
train_imgs = np.stack([inst['image'] for inst in data['train']], axis=0)
train_lbls = np.array([inst['label'] for inst in data['train']])
test_imgs  = np.stack([inst['image'] for inst in data['test']],  axis=0)
test_lbls  = np.array([inst['label'] for inst in data['test']])

# 4. Convert to PyTorch tensors and normalize to [0,1]:
#    Also, move channel dim to front: (N, C, H, W)
def to_tensor(imgs):
    """
    Convert a NumPy array of images to a float32 PyTorch tensor
    normalized to [0,1], with shape (N, 1, H, W).
    Handles:
      - imgs: (N, H, W, 3)   → single grayscale channel
      - imgs: (N, H, W)      → single grayscale channel
    """
    imgs = torch.tensor(imgs, dtype=torch.float32) / 255.0

    # If it's (N, H, W, 3), drop the last (identical) channels
    if imgs.ndim == 4 and imgs.shape[-1] == 3:
        imgs = imgs[..., 0]         # now (N, H, W)

    # If it's (N, H, W), add the channel dimension
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)    # (N, 1, H, W)

    return imgs


train_X = to_tensor(train_imgs)
train_y = torch.tensor(train_lbls, dtype=torch.long)
test_X  = to_tensor(test_imgs)
test_y  = torch.tensor(test_lbls,  dtype=torch.long)

# 5. Wrap in TensorDataset and DataLoader:
batch_size = 16
train_ds = TensorDataset(train_X, train_y)
test_ds  = TensorDataset(test_X,  test_y)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size)

import torch.nn as nn
import torch.nn.functional as F

class ORLNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1 input channel (grayscale), 16 filters, 3×3 kernels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # After two poolings, H and W are each divided by 4:
        # ORL images are 112×92 → ~28×23
        self.fc1   = nn.Linear(32 * 28 * 23, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # halves H,W
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # halves again
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

if __name__ == '__main__':
    import torch
    import torch_directml   # or use torch.device('cuda') if on CUDA

    # 1) Device
    device = torch_directml.device() if not torch.cuda.is_available() else torch.device('cuda')

    # 2) Number of classes
    num_classes = len(np.unique(train_lbls))

    # 3) Instantiate model, optimizer, and loss
    model     = ORLNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 4) Training loop
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # compute full metrics on train & test
        train_stats = evaluate_full(model, train_loader, criterion, device)
        test_stats = evaluate_full(model, test_loader, criterion, device)

        # print summary in decimals
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train — loss: {train_stats['loss']:.4f}, "
              f"acc: {train_stats['accuracy']:.4f}")
        for avg in ("micro", "macro", "weighted"):
            print(f"    {avg:>8} — precision: {train_stats[f'precision_{avg}']:.4f}, "
                  f"recall:    {train_stats[f'recall_{avg}']:.4f}, "
                  f"f1:        {train_stats[f'f1_{avg}']:.4f}")

        print(f"  Test  — loss: {test_stats['loss']:.4f}, "
              f"acc: {test_stats['accuracy']:.4f}")
        for avg in ("micro", "macro", "weighted"):
            print(f"    {avg:>8} — precision: {test_stats[f'precision_{avg}']:.4f}, "
                  f"recall:    {test_stats[f'recall_{avg}']:.4f}, "
                  f"f1:        {test_stats[f'f1_{avg}']:.4f}")

        print("-" * 60)