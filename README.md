
# AI Face Emotion Recognition 🎭

![result-1773258444859](https://github.com/user-attachments/assets/60e1d19c-cf60-44c9-b904-4f69e2b7e61d)

> **Olympiad Project — Dana Stukalova**

![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FER-2013](https://img.shields.io/badge/Dataset-FER--2013-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🗺️ Project Map

```
┌──────────────────────────────────────────────┐
│  AI FACE EMOTION RECOGNITION (Olympiad)      │
│  MODE : PyQt5 GUI • PyTorch • FER-2013       │
└──────────────────────────────────────────────┘

Camera → Face Detection → Emotion Model → GUI
```

## 🛠️ Tech Stack

- **Backend:** Python 3.11, PyTorch 2.1, torchvision
- **GUI:** PyQt5
- **Data:** FER-2013 (7 emotions)
- **Visualization:** Matplotlib, scikit-learn
- **Packaging:** PyInstaller (.exe build)

## ✨ Key Features

- Real-time emotion recognition from webcam
- MobileNetV2 / ResNet18 (transfer learning)
- Custom augmentations for FER-2013
- GUI: camera selection, resolution, confidence threshold
- Blind Test mode (folder/image)
- Results viewer: confusion matrix, ROC, training curves, block diagram
- Cyrillic path safety (Windows)
- Offline install (wheels, no internet required)

## 🚀 Quickstart

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Data Preparation

Download FER-2013 and place in:
```
tools/downloads/train/angry/*.jpg, happy/*.jpg, ...
tools/downloads/test/angry/*.jpg, happy/*.jpg, ...
```

### 3. Training

```bash
python train.py --model mobilenet --epochs 40 --batch 64 --lr 0.0003
```

### 4. Run GUI

```bash
python app.py
```

### 5. Inference (CLI)

```bash
python infer.py
python infer.py --image photo.jpg
```

## 🏗️ Architecture

```
┌───────────────┐
│  Webcam       │
└─────┬─────────┘
    │
┌─────▼─────────┐
│  Face Detect  │
│  (Haar Cascade│
└─────┬─────────┘
    │
┌─────▼─────────┐
│  Preprocess   │
│  (Resize,     │
│  Normalize)   │
└─────┬─────────┘
    │
┌─────▼─────────┐
│  Emotion Model│
│  (MobileNetV2 │
│  / ResNet18)  │
└─────┬─────────┘
    │
┌─────▼─────────┐
│  GUI / Output │
└───────────────┘
```

## 📊 Results

- Accuracy, F1, confusion matrix, ROC, training curves
- Results saved in `results/`

## 🧠 Model Details

- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Fine-tuning:** features.10–18 unfrozen
- **Classifier:** Dropout(0.4) → Linear(1280 → 7)
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=5, T_mult=2)
- **Loss:** CrossEntropyLoss (label_smoothing=0.1, class weights)
- **Early stopping:** patience=8

## 📁 Project Structure

```
ml_contest/
├── app.py                  # PyQt5 GUI
├── train.py                # Model training
├── infer.py                # CLI inference
├── generate_diagram.py     # Block diagram
├── build_exe.py            # PyInstaller build
├── runtime_hook_dlls.py    # DLL path fix for .exe
├── requirements.txt        # Dependencies
├── run_demo.bat            # Menu launcher
├── models/                 # Model weights (.gitignore)
│   ├── best_model.pt
│   ├── mobilenet_v2_pretrained.pt
│   └── resnet18_pretrained.pt
├── results/                # Metrics, plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   ├── block_diagram.png
│   └── metrics.json
├── utils/
│   ├── dataset.py          # FER-2013 loader
│   └── metrics.py          # Metrics, plots
└── tools/
    ├── downloads/         # FER-2013 data
    └── fetch_pretrained.py # Pretrained weights
```

## 🎯 Project Goals

- Practice transfer learning and PyQt5 GUI
- Robust Windows support (Cyrillic paths)
- Reproducible offline install
- Clean code for GitHub

## 👤 Author

**Darya Stukalova** — Python, AI, Olympiad

---

> _Educational project for AI Olympiad. All code is original and cleaned for GitHub._
