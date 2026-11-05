"""Fine-tune a ResNet-18 garment classifier (upper / lower / dress).

Usage:
    python backend/train_classifier.py --epochs 12 --batch-size 32
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "garments"
MODELS_DIR = ROOT / "models"
CLASSES = ["upper", "lower", "dress"]
IMG_SIZE = 224


class GarmentDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_samples():
    by_class = defaultdict(list)
    for label, name in enumerate(CLASSES):
        for p in sorted((DATA_DIR / name).glob("*.jpg")):
            by_class[name].append((p, label))
    train, val = [], []
    rng = random.Random(42)
    for name, items in by_class.items():
        rng.shuffle(items)
        cut = int(len(items) * 0.8)
        train.extend(items[:cut])
        val.extend(items[cut:])
    return train, val, by_class


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_samples, val_samples, by_class = build_samples()
    counts = {k: len(v) for k, v in by_class.items()}
    print(f"class counts: {counts}  train={len(train_samples)} val={len(val_samples)}", flush=True)
    if not train_samples:
        raise SystemExit("No training images found. Run data_prep.py first.")

    train_ds = GarmentDataset(train_samples, train_tf)
    val_ds = GarmentDataset(val_samples, val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    class_counts = torch.tensor([counts[c] for c in CLASSES], dtype=torch.float32)
    class_weights = (class_counts.sum() / (len(CLASSES) * class_counts)).to(device)
    print(f"class weights: {class_weights.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "garment_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running / len(train_ds)

        model.eval()
        correct, total = 0, 0
        cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                for t, p in zip(labels.cpu(), preds.cpu()):
                    cm[t, p] += 1
        acc = correct / max(total, 1)
        print(f"epoch {epoch:02d}/{args.epochs}  loss={train_loss:.4f}  val_acc={acc:.3f}", flush=True)

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "img_size": IMG_SIZE,
                    "val_acc": acc,
                },
                ckpt_path,
            )

    (MODELS_DIR / "garment_classes.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")

    print(f"best val accuracy: {best_acc:.3f}", flush=True)
    print("confusion matrix (rows=true, cols=pred):", flush=True)
    print("labels:", CLASSES, flush=True)
    for i, row in enumerate(cm.tolist()):
        print(f"  {CLASSES[i]:>6}: {row}", flush=True)

    for i, c in enumerate(CLASSES):
        tp = cm[i, i].item()
        fn = cm[i].sum().item() - tp
        fp = cm[:, i].sum().item() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        print(f"  {c:>6}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}", flush=True)

    print(f"saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
