"""
FER-2013 Training Script
========================
Trains a custom CNN (EmotionNet) on the FER-2013 dataset and exports
the best checkpoint directly to fer.onnx for deployment.

Dataset layout expected at  <project_root>/dataset/
    dataset/
        train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
        test/   angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

Usage (run from project root):
    pip install torch torchvision
    python FacialEmotionRecognition/train.py
    python FacialEmotionRecognition/train.py --epochs 80 --batch-size 64
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Paths (relative to this file) ─────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, '..', 'dataset')
CHECKPOINT = os.path.join(_HERE, 'fer_best.pth')
ONNX_OUT   = os.path.join(_HERE, 'fer.onnx')

NUM_CLASSES = 7
IMG_SIZE    = 48

# Alphabetical order — matches torchvision.datasets.ImageFolder
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# ── Architecture ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two conv layers with BN + ReLU, followed by MaxPool and Dropout."""
    def __init__(self, in_ch, out_ch, drop=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        return self.block(x)


class EmotionNet(nn.Module):
    """
    Custom CNN tuned for 48×48 grayscale emotion classification.

    Spatial progression:  48 → 24 → 12 → 6 → 1  (via AdaptiveAvgPool)
    Channel progression:  1  → 64 → 128 → 256 → 512
    """
    def __init__(self, num_classes: int = NUM_CLASSES, drop_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   64,  drop=0.25),   # → (64,  24, 24)
            ConvBlock(64,  128, drop=0.25),   # → (128, 12, 12)
            ConvBlock(128, 256, drop=0.25),   # → (256,  6,  6)

            # Final block: no pool, just two convs then global avg pool
            # nn.Conv2d(256, 512, 3, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d(1),          # → (512, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Data ──────────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomCrop(IMG_SIZE, padding=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_tf, val_tf


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def per_class_accuracy(model, loader, device):
    model.eval()
    correct = torch.zeros(NUM_CLASSES)
    counts  = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for c in range(NUM_CLASSES):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                counts[c]  += mask.sum().item()
    return {LABELS[i]: (correct[i] / counts[i].clamp(min=1)).item() for i in range(NUM_CLASSES)}


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, device):
    model.eval()
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model, dummy, ONNX_OUT,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=18,
    )
    print(f"\n✓  ONNX model exported  →  {ONNX_OUT}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train EmotionNet on FER-2013')
    parser.add_argument('--epochs',     type=int,   default=80)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--workers',    type=int,   default=4)
    args = parser.parse_args()

    # ── Device ──
    device = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice     : {device}")
    print(f"Epochs     : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

    # ── Datasets ──
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  transform=val_tf)

    print(f"Classes    : {train_ds.classes}")
    print(f"Train      : {len(train_ds):,} images")
    print(f"Val        : {len(val_ds):,} images\n")

    # Class weights — inverse frequency to handle disgust imbalance
    counts  = torch.zeros(NUM_CLASSES)
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    weights = (counts.sum() / (NUM_CLASSES * counts)).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model / loss / optimiser ──
    model     = EmotionNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warm up for 5 epochs, then cosine anneal
    warmup    = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5, eta_min=1e-5)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {total_params:,}\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'':>8}")
    print('─' * 72)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        flag = ''
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), CHECKPOINT)
            flag = '← best'

        lr_now = optimizer.param_groups[0]['lr']
        print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:9.3%}  "
              f"{vl_loss:8.4f}  {vl_acc:7.3%}  {lr_now:8.2e}  {flag}")

    # ── Final evaluation ──
    print(f"\n{'─'*72}")
    print(f"Best val accuracy : {best_acc:.3%}\n")

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    acc_per_class = per_class_accuracy(model, val_loader, device)
    print("Per-class accuracy (best model):")
    for cls, acc in acc_per_class.items():
        bar = '█' * int(acc * 30)
        print(f"  {cls:<10} {acc:.1%}  {bar}")

    # ── Export ──
    export_onnx(model, device)
    print("\nCommit fer.onnx to deploy the new model.")


if __name__ == '__main__':
    main()
