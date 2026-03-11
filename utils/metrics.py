"""
Метрики: Confusion Matrix, ROC-кривая, F1 score.
Критерий Оптимизация — построение матрицы ошибок, ROC, F1.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, auc, accuracy_score
)
from sklearn.preprocessing import label_binarize


def compute_and_save_metrics(y_true, y_pred, y_probs, class_names, output_dir='results'):
    """
    Вычисляет и сохраняет все метрики в output_dir:
      - metrics.json       (accuracy, F1, per-class)
      - confusion_matrix.png
      - roc_curve.png

    Args:
        y_true:  list/array истинных меток (int)
        y_pred:  list/array предсказанных меток (int)
        y_probs: np.array softmax-вероятностей (N x num_classes)
        class_names: list имён классов
        output_dir: папка для сохранения
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    metrics = {
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_per_class': {c: round(f, 4) for c, f in zip(class_names, f1_per_class)},
        'classification_report': report,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[metrics] Accuracy={acc:.4f}  F1(macro)={f1_macro:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True', xlabel='Predicted',
           title=f'Confusion Matrix  (acc={acc:.3f})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)
    fig.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] Confusion matrix saved: {cm_path}")

    if y_probs is not None and len(y_probs) > 0:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        for i in range(num_classes):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set(xlabel='FPR', ylabel='TPR', title='ROC Curve (one-vs-rest)')
        ax2.legend(fontsize=7)
        fig2.tight_layout()
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        fig2.savefig(roc_path, dpi=150)
        plt.close(fig2)
        print(f"[metrics] ROC curve saved: {roc_path}")

    return metrics


def plot_training_curves(history, output_dir='results'):
    """
    Строит графики Loss и Accuracy по эпохам.
    history = [{'epoch': 1, 'train_loss': ..., 'train_acc': ..., 'val_loss': ..., 'val_acc': ...}, ...]
    Сохраняет training_curves.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_loss, 'r-o', label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss по эпохам')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, 'b-o', label='Train Acc', markersize=4)
    ax2.plot(epochs, val_acc, 'r-o', label='Val Acc', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy по эпохам')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[metrics] Training curves saved: {path}")
    return path
