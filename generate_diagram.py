"""
generate_diagram.py — Генерация блок-схемы проекта (matplotlib).
Создаёт файл results/block_diagram.png

Использование:
    python generate_diagram.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color='#2196F3', text_color='white', fontsize=10, style='round'):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.15" if style == 'round' else f"square,pad=0.1",
                          facecolor=color, edgecolor='#263238', linewidth=1.5,
                          zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=4, wrap=True,
            fontfamily='sans-serif')


def draw_arrow(ax, x1, y1, x2, y2, color='#455A64'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                connectionstyle='arc3,rad=0'),
                zorder=2)


def draw_label(ax, x, y, text, fontsize=8, color='#607D8B'):
    """Draw a small label."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontstyle='italic', zorder=5)


def generate_main_pipeline():
    """Основная блок-схема пайплайна распознавания эмоций."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 22))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 25)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    ax.text(5, 24.2, 'БЛОК-СХЕМА ПРОЕКТА', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#1A237E', fontfamily='sans-serif')
    ax.text(5, 23.6, 'Распознавание эмоций в видеопотоке (FER-2013)',
            ha='center', va='center', fontsize=13, color='#455A64', fontfamily='sans-serif')

    ax.text(2.5, 22.5, '① ДАННЫЕ', ha='center', fontsize=14, fontweight='bold', color='#E65100')

    draw_box(ax, 2.5, 21.5, 3.5, 0.8, 'FER-2013 Dataset\n35 887 изображений', '#FF9800', 'white', 10)
    draw_arrow(ax, 2.5, 21.1, 2.5, 20.4)

    draw_box(ax, 1.2, 19.9, 2.0, 0.7, 'Train\n28 709', '#FFA726', 'white', 9)
    draw_box(ax, 3.8, 19.9, 2.0, 0.7, 'Test\n7 178', '#FFA726', 'white', 9)
    draw_arrow(ax, 2.0, 21.1, 1.2, 20.3)
    draw_arrow(ax, 3.0, 21.1, 3.8, 20.3)

    draw_arrow(ax, 1.2, 19.55, 1.2, 18.9)

    draw_box(ax, 1.2, 18.5, 2.2, 0.7, 'Аугментации', '#43A047', 'white', 9)
    ax.text(3.0, 18.8, '• Flip, Rotate ±15°', fontsize=7, color='#2E7D32')
    ax.text(3.0, 18.5, '• ColorJitter, Blur', fontsize=7, color='#2E7D32')
    ax.text(3.0, 18.2, '• Perspective, Erasing', fontsize=7, color='#2E7D32')

    draw_arrow(ax, 1.2, 18.15, 1.2, 17.5)

    draw_box(ax, 2.5, 17.1, 3.5, 0.7, 'Resize 128×128 → RGB → Normalize', '#66BB6A', 'white', 9)
    draw_arrow(ax, 1.2, 18.15, 2.5, 17.5)

    ax.text(7.5, 22.5, '② МОДЕЛЬ', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

    draw_box(ax, 7.5, 21.5, 3.5, 0.8, 'ImageNet Pretrained\n14M изображений, 1000 классов', '#1565C0', 'white', 9)
    draw_arrow(ax, 7.5, 21.1, 7.5, 20.4)

    draw_box(ax, 7.5, 19.9, 3.8, 0.8, 'MobileNetV2\nDepthwise Separable Conv\nInverted Residuals', '#1E88E5', 'white', 9)
    draw_arrow(ax, 7.5, 19.5, 7.5, 18.9)

    draw_box(ax, 6.2, 18.5, 2.0, 0.7, 'features.0-9\nFROZEN', '#90CAF9', '#1A237E', 8)
    draw_box(ax, 8.8, 18.5, 2.0, 0.7, 'features.10-18\nUNFROZEN', '#EF5350', 'white', 8)
    draw_arrow(ax, 7.0, 19.5, 6.2, 18.9)
    draw_arrow(ax, 8.0, 19.5, 8.8, 18.9)

    draw_arrow(ax, 8.8, 18.15, 8.8, 17.5)

    draw_box(ax, 7.5, 17.1, 3.5, 0.7, 'Dropout(0.4) → Linear(1280 → 7)', '#7E57C2', 'white', 10)
    draw_arrow(ax, 8.8, 18.15, 7.5, 17.5)

    ax.text(5, 16.0, '③ ОБУЧЕНИЕ', ha='center', fontsize=14, fontweight='bold', color='#6A1B9A')

    draw_arrow(ax, 2.5, 16.7, 5, 15.5)
    draw_arrow(ax, 7.5, 16.7, 5, 15.5)

    draw_box(ax, 5, 15.1, 4.0, 0.7, 'Forward Pass: images → model → predictions', '#7E57C2', 'white', 9)
    draw_arrow(ax, 5, 14.75, 5, 14.2)

    draw_box(ax, 5, 13.8, 4.0, 0.7, 'CrossEntropyLoss\n(label smoothing=0.1, class weights)', '#AB47BC', 'white', 9)
    draw_arrow(ax, 5, 13.45, 5, 12.9)

    draw_box(ax, 5, 12.5, 4.0, 0.7, 'Backward → Gradient Clipping → AdamW', '#CE93D8', '#1A237E', 9)
    draw_arrow(ax, 5, 12.15, 5, 11.6)

    draw_box(ax, 5, 11.2, 4.0, 0.7, 'CosineAnnealingWarmRestarts\nlr: 0.0005 → 1e-6 → restart', '#9C27B0', 'white', 9)
    draw_arrow(ax, 5, 10.85, 5, 10.3)

    draw_box(ax, 5, 9.9, 3.5, 0.7, 'Epoch 1..25 | Early Stop (patience=8)', '#E91E63', 'white', 9)

    ax.annotate('', xy=(1.5, 15.1), xytext=(1.5, 9.9),
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5,
                                connectionstyle='arc3,rad=-0.3'),
                zorder=2)
    ax.text(0.6, 12.5, 'repeat\nepochs', fontsize=8, color='#E91E63', ha='center', rotation=90)

    draw_arrow(ax, 5, 9.55, 5, 8.9)

    draw_box(ax, 5, 8.5, 3.5, 0.7, 'Save best_model.pt\n(if val_acc improved)', '#4CAF50', 'white', 9)

    ax.text(8.5, 8.0, '④ МЕТРИКИ', ha='center', fontsize=14, fontweight='bold', color='#00695C')

    draw_arrow(ax, 6.5, 8.15, 8.5, 7.5)

    draw_box(ax, 8.5, 7.1, 2.5, 0.6, 'Confusion Matrix', '#00897B', 'white', 9)
    draw_box(ax, 8.5, 6.3, 2.5, 0.6, 'ROC Curve / AUC', '#00897B', 'white', 9)
    draw_box(ax, 8.5, 5.5, 2.5, 0.6, 'F1-score / Accuracy', '#00897B', 'white', 9)
    draw_box(ax, 8.5, 4.7, 2.5, 0.6, 'Training Curves', '#00897B', 'white', 9)

    draw_arrow(ax, 8.5, 6.8, 8.5, 6.6)
    draw_arrow(ax, 8.5, 6.0, 8.5, 5.8)
    draw_arrow(ax, 8.5, 5.2, 8.5, 5.0)

    ax.text(2.5, 8.0, '⑤ ИНФЕРЕНС', ha='center', fontsize=14, fontweight='bold', color='#BF360C')

    draw_arrow(ax, 3.5, 8.15, 2.5, 7.5)

    draw_box(ax, 2.5, 7.1, 2.8, 0.6, 'Веб-камера', '#FF5722', 'white', 10)
    draw_arrow(ax, 2.5, 6.8, 2.5, 6.3)

    draw_box(ax, 2.5, 5.9, 2.8, 0.6, 'OpenCV: захват кадра', '#FF7043', 'white', 9)
    draw_arrow(ax, 2.5, 5.6, 2.5, 5.1)

    draw_box(ax, 2.5, 4.7, 2.8, 0.6, 'Haar Cascade\nдетекция лиц', '#FF8A65', 'white', 9)
    draw_arrow(ax, 2.5, 4.4, 2.5, 3.9)

    draw_box(ax, 2.5, 3.5, 2.8, 0.6, 'Вырезать лицо\nResize → Normalize', '#FFAB91', '#3E2723', 9)
    draw_arrow(ax, 2.5, 3.2, 2.5, 2.7)

    draw_box(ax, 2.5, 2.3, 2.8, 0.6, 'MobileNetV2\nforward pass', '#1E88E5', 'white', 9)
    draw_arrow(ax, 2.5, 2.0, 2.5, 1.5)

    draw_box(ax, 2.5, 1.1, 2.8, 0.6, 'Softmax → 7 вероятностей', '#7E57C2', 'white', 9)
    draw_arrow(ax, 2.5, 0.8, 2.5, 0.3)

    draw_box(ax, 2.5, -0.1, 2.8, 0.6, 'Эмоция + Рамка\nна кадре', '#4CAF50', 'white', 9)

    ax.text(5.8, 3.8, '⑥ GUI', ha='center', fontsize=14, fontweight='bold', color='#283593')

    draw_box(ax, 5.8, 3.1, 2.5, 0.6, 'PyQt5 GUI', '#3F51B5', 'white', 10)
    draw_arrow(ax, 5.8, 2.8, 5.8, 2.3)

    draw_box(ax, 5.8, 1.9, 2.8, 0.6, '• Камера + эмоции\n• Blind Test\n• Просмотр метрик', '#5C6BC0', 'white', 8)
    draw_arrow(ax, 5.8, 1.6, 5.8, 1.0)

    draw_box(ax, 5.8, 0.6, 2.5, 0.6, 'Демо жюри', '#283593', 'white', 10)

    draw_arrow(ax, 3.9, -0.1, 5.8, 0.3)

    legend_y = -0.7
    ax.text(0, legend_y, 'Цвета:', fontsize=9, fontweight='bold', color='#263238')
    colors_legend = [
        ('#FF9800', 'Данные'), ('#1E88E5', 'Модель'), ('#7E57C2', 'Обучение'),
        ('#00897B', 'Метрики'), ('#FF5722', 'Инференс'), ('#3F51B5', 'GUI'),
    ]
    for i, (c, label) in enumerate(colors_legend):
        bx = 1.8 + i * 1.6
        box = FancyBboxPatch((bx - 0.3, legend_y - 0.15), 0.6, 0.3,
                              boxstyle="round,pad=0.05", facecolor=c, edgecolor='none', zorder=3)
        ax.add_patch(box)
        ax.text(bx + 0.5, legend_y, label, fontsize=8, va='center', color='#263238')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'block_diagram.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close(fig)
    print(f'[diagram] Saved: {out_path}')
    return out_path


if __name__ == '__main__':
    generate_main_pipeline()
