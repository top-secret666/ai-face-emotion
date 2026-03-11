"""
train.py — Обучение модели на FER-2013 (transfer learning).
Покрывает критерии: Оптимизация (аугментация, fine-tuning, accuracy, F1, confusion matrix, ROC).

Использование:
    python train.py --train_dir tools/downloads/train --test_dir tools/downloads/test \
                    --model mobilenet --epochs 40 --batch 64 --lr 0.0003 \
                    --save_name best_model.pt
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_dataloaders, EMOTION_CLASSES, NUM_CLASSES
from utils.metrics import compute_and_save_metrics, plot_training_curves


def build_model(arch='mobilenet', num_classes=NUM_CLASSES, pretrained_path=None, unfreeze_all=False):
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        if pretrained_path and os.path.isfile(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"[train] Loaded pretrained ResNet18 from {pretrained_path}")
        else:
            try:
                model = models.resnet18(weights='IMAGENET1K_V1')
                print("[train] Loaded ResNet18 IMAGENET1K_V1 weights")
            except Exception:
                print("[train] Using ResNet18 without pretrained weights")
        if unfreeze_all:
            for param in model.parameters():
                param.requires_grad = True
            print("[train] All layers unfrozen (full fine-tuning)")
        else:
            for name, param in model.named_parameters():
                if not any(k in name for k in ['layer2', 'layer3', 'layer4', 'fc']):
                    param.requires_grad = False
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    else:
        model = models.mobilenet_v2(weights=None)
        if pretrained_path and os.path.isfile(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"[train] Loaded pretrained MobileNetV2 from {pretrained_path}")
        else:
            try:
                model = models.mobilenet_v2(weights='IMAGENET1K_V1')
                print("[train] Loaded MobileNetV2 IMAGENET1K_V1 weights")
            except Exception:
                print("[train] Using MobileNetV2 without pretrained weights")
        if unfreeze_all:
            for param in model.parameters():
                param.requires_grad = True
            print("[train] All layers unfrozen (full fine-tuning)")
        else:
            for name, param in model.named_parameters():
                if not any(k in name for k in ['features.10', 'features.11', 'features.12',
                                                'features.13', 'features.14', 'features.15',
                                                'features.16', 'features.17', 'features.18',
                                                'classifier']):
                    param.requires_grad = False
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.last_channel, num_classes))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    all_true, all_pred, all_probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    return total_loss / total, correct / total, all_true, all_pred, all_probs


def main():
    parser = argparse.ArgumentParser(description='FER-2013 Training')
    parser.add_argument('--train_dir', type=str, default='tools/downloads/train')
    parser.add_argument('--test_dir', type=str, default='tools/downloads/test')
    parser.add_argument('--model', type=str, default='mobilenet', choices=['resnet18', 'mobilenet'])
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--save_name', type=str, default='best_model.pt')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--unfreeze_all', action='store_true',
                        help='Unfreeze all layers for full fine-tuning (slower but better accuracy)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    base = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base, args.train_dir)
    test_dir = os.path.join(base, args.test_dir)
    results_dir = os.path.join(base, args.results_dir)
    models_dir = os.path.join(base, 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    pretrained_path = args.pretrained
    if pretrained_path is None:
        if args.model == 'mobilenet':
            pretrained_path = os.path.join(models_dir, 'mobilenet_v2_pretrained.pt')
        else:
            pretrained_path = os.path.join(models_dir, 'resnet18_pretrained.pt')

    print(f"[train] Loading data: train={train_dir}, test={test_dir}")
    train_loader, test_loader = get_dataloaders(
        train_dir, test_dir, batch_size=args.batch, img_size=args.img_size
    )
    print(f"[train] Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = build_model(args.model, NUM_CLASSES, pretrained_path, unfreeze_all=args.unfreeze_all).to(device)
    class_counts = []
    for cls_name in EMOTION_CLASSES:
        cls_dir = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_dir):
            class_counts.append(len(os.listdir(cls_dir)))
        else:
            class_counts.append(1)
    total_samples = sum(class_counts)
    class_weights = torch.tensor([total_samples / (NUM_CLASSES * c) for c in class_counts],
                                  dtype=torch.float32).to(device)
    print(f"[train] Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    best_acc = 0.0
    no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={elapsed:.1f}s")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            save_path = os.path.join(models_dir, args.save_name)
            torch.save(model.state_dict(), save_path)
            print(f"  * Best model saved: {save_path} (acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= 8:
                print(f"  [early stop] No improvement for {no_improve} epochs, stopping.")
                break

    best_path = os.path.join(models_dir, args.save_name)
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, final_acc, y_true, y_pred, y_probs = evaluate(model, test_loader, device)
    compute_and_save_metrics(y_true, y_pred, y_probs, EMOTION_CLASSES, results_dir)
    print(f"\n[train] Done. Best accuracy: {best_acc:.4f}  Results: {results_dir}/")

    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, results_dir)

    try:
        import tempfile, shutil
        model.eval()
        dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        traced = torch.jit.trace(model, dummy)
        jit_name = args.save_name.replace('.pt', '.jit.pt')
        jit_path = os.path.join(models_dir, jit_name)
        tmp = os.path.join(tempfile.gettempdir(), jit_name)
        traced.save(tmp)
        shutil.copy2(tmp, jit_path)
        os.remove(tmp)
        print(f"[train] TorchScript model saved: {jit_path}")
    except Exception as e:
        print(f"[train] TorchScript export failed: {e}")


if __name__ == '__main__':
    main()
