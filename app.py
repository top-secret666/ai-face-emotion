"""
app.py — PyQt5 GUI для олимпиады.

Задание 1: Распознавание эмоций в видеопотоке (FER-2013, transfer learning)

Использование:   python app.py
"""
import os
import sys
import shutil
import tempfile
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

def _fix_qt_plugins():
    """Copy Qt platform plugins to temp dir if path contains non-ASCII chars."""
    try:
        import PyQt5
        qt_dir = os.path.dirname(PyQt5.__file__)
        for subdir in ('Qt5', 'Qt'):
            platforms_src = os.path.join(qt_dir, subdir, 'plugins', 'platforms')
            if os.path.isdir(platforms_src):
                try:
                    platforms_src.encode('ascii')
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platforms_src
                except UnicodeEncodeError:
                    tmp = os.path.join(tempfile.gettempdir(), 'qt5_platforms')
                    if not os.path.isdir(tmp) or not os.listdir(tmp):
                        if os.path.isdir(tmp):
                            shutil.rmtree(tmp)
                        shutil.copytree(platforms_src, tmp)
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = tmp
                return
    except Exception:
        pass

_fix_qt_plugins()

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                  QHBoxLayout, QLabel, QPushButton, QComboBox,
                                  QFileDialog, QGroupBox, QStatusBar,
                                  QMessageBox, QDialog, QScrollArea, QTextEdit)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap, QFont
except ImportError:
    print('PyQt5 not installed. Run: pip install PyQt5')
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import EMOTION_CLASSES, NUM_CLASSES

if getattr(sys, 'frozen', False):
    BASE = sys._MEIPASS
else:
    BASE = os.path.dirname(os.path.abspath(__file__))


def _safe_path(path):
    """If path contains non-ASCII chars, copy file to temp and return safe path.
    This works around OpenCV / Qt not handling Cyrillic paths on Windows."""
    try:
        path.encode('ascii')
        return path
    except UnicodeEncodeError:
        pass
    import hashlib
    name_hash = hashlib.md5(path.encode('utf-8')).hexdigest()[:8]
    ext = os.path.splitext(path)[1]
    tmp = os.path.join(tempfile.gettempdir(), f'mlcontest_{name_hash}{ext}')
    if not os.path.isfile(tmp) or os.path.getmtime(path) > os.path.getmtime(tmp):
        shutil.copy2(path, tmp)
    return tmp


def _safe_imread(path):
    """cv2.imread that handles non-ASCII (Cyrillic) file paths on Windows."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


EMOTION_RU = {
    'angry': 'Злость', 'disgust': 'Отвращение', 'fear': 'Страх',
    'happy': 'Радость', 'neutral': 'Нейтрально', 'sad': 'Грусть',
    'surprise': 'Удивление',
}

EMOTION_COLORS = {
    'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128),
    'happy': (0, 255, 255), 'neutral': (200, 200, 200),
    'sad': (255, 0, 0), 'surprise': (0, 165, 255),
}

class ImageViewerDialog(QDialog):
    """Dialog to view large images with zoom controls and fit-to-window."""
    def __init__(self, parent, image_path, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 800)
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        btn_zoom_in = QPushButton('Zoom +')
        btn_zoom_out = QPushButton('Zoom -')
        btn_fit = QPushButton('По окну')
        btn_actual = QPushButton('100%')
        toolbar.addWidget(btn_zoom_in)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_fit)
        toolbar.addWidget(btn_actual)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.img_label)
        layout.addWidget(self.scroll)

        btn_close = QPushButton('Закрыть')
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

        safe = _safe_path(image_path)
        self.pixmap = QPixmap(safe)
        self.scale = 1.0
        if self.pixmap.isNull():
            self.img_label.setText(f'Не удалось загрузить: {image_path}')
            return

        btn_zoom_in.clicked.connect(lambda: self._zoom(1.25))
        btn_zoom_out.clicked.connect(lambda: self._zoom(0.8))
        btn_actual.clicked.connect(lambda: self._set_scale(1.0))
        btn_fit.clicked.connect(self._fit_to_window)

        self._fit_to_window()

    def _update_pixmap(self):
        if self.pixmap.isNull():
            return
        new_w = max(1, int(self.pixmap.width() * self.scale))
        new_h = max(1, int(self.pixmap.height() * self.scale))
        scaled = self.pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled)
        self.img_label.adjustSize()

    def _zoom(self, factor):
        self.scale *= factor
        self.scale = max(0.05, min(self.scale, 10.0))
        self._update_pixmap()

    def _set_scale(self, scale):
        self.scale = scale
        self._update_pixmap()

    def _fit_to_window(self):
        if self.pixmap.isNull():
            return
        vw = self.scroll.viewport().width() or 800
        vh = self.scroll.viewport().height() or 600
        scale_w = vw / self.pixmap.width()
        scale_h = vh / self.pixmap.height()
        self.scale = min(scale_w, scale_h, 1.0)
        self._update_pixmap()


def build_emotion_model(arch, weights_path, device):
    if arch == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, NUM_CLASSES))
    else:
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.last_channel, NUM_CLASSES))
    if weights_path and os.path.isfile(weights_path):
        m.load_state_dict(torch.load(weights_path, map_location=device))
    m.to(device).eval()
    return m


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Olympiad — Стукалова')
        self.setMinimumSize(960, 700)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.face_cascade = self._load_face_cascade()
        self.img_size = 128
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.fps_history = []
        self.current_arch = 'mobilenet'

        self._build_ui()

    @staticmethod
    def _load_face_cascade():
        """Load Haar cascade, handling Cyrillic/non-ASCII paths."""
        cascade_name = 'haarcascade_frontalface_default.xml'
        cascade_path = cv2.data.haarcascades + cascade_name
        safe = _safe_path(cascade_path)
        cascade = cv2.CascadeClassifier(safe)
        if cascade.empty():
            print('[app] WARNING: Could not load Haar cascade from', safe)
        return cascade

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        ctrl = QHBoxLayout()
        self.cam_combo = QComboBox()
        for i in range(5):
            self.cam_combo.addItem(f'Camera {i}')
        ctrl.addWidget(QLabel('Камера:'))
        ctrl.addWidget(self.cam_combo)

        self.res_combo = QComboBox()
        self.res_combo.addItems(['640x480', '800x600', '1280x720', '1920x1080'])
        ctrl.addWidget(QLabel('Разрешение:'))
        ctrl.addWidget(self.res_combo)

        self.arch_combo = QComboBox()
        self.arch_combo.addItems(['MobileNetV2', 'ResNet18'])
        ctrl.addWidget(QLabel('Архитектура:'))
        ctrl.addWidget(self.arch_combo)

        self.btn_load = QPushButton('Загрузить модель')
        self.btn_load.clicked.connect(self._load_emotion_model)
        ctrl.addWidget(self.btn_load)

        self.btn_start = QPushButton('▶ Старт')
        self.btn_start.clicked.connect(self._toggle_camera)
        ctrl.addWidget(self.btn_start)

        self.btn_blind = QPushButton('Blind Test ▶')
        self.btn_blind.clicked.connect(self._blind_test)
        ctrl.addWidget(self.btn_blind)
        root.addLayout(ctrl)

        results_row = QHBoxLayout()
        self.btn_show_cm = QPushButton('📊 Confusion Matrix')
        self.btn_show_cm.clicked.connect(lambda: self._show_results_image('confusion_matrix.png', 'Матрица ошибок'))
        results_row.addWidget(self.btn_show_cm)

        self.btn_show_roc = QPushButton('📈 ROC-кривая')
        self.btn_show_roc.clicked.connect(lambda: self._show_results_image('roc_curve.png', 'ROC-кривая'))
        results_row.addWidget(self.btn_show_roc)

        self.btn_show_curves = QPushButton('📉 Кривые обучения')
        self.btn_show_curves.clicked.connect(lambda: self._show_results_image('training_curves.png', 'Кривые обучения'))
        results_row.addWidget(self.btn_show_curves)

        self.btn_show_diagram = QPushButton('🗒 Блок-схема')
        self.btn_show_diagram.clicked.connect(lambda: self._show_results_image('block_diagram.png', 'Блок-схема проекта'))
        results_row.addWidget(self.btn_show_diagram)

        self.btn_show_metrics = QPushButton('📋 metrics.json')
        self.btn_show_metrics.clicked.connect(self._show_metrics_json)
        results_row.addWidget(self.btn_show_metrics)
        root.addLayout(results_row)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet('background: black;')
        root.addWidget(self.video_label)

        info = QHBoxLayout()
        self.lbl_fps = QLabel('FPS: —')
        self.lbl_fps.setFont(QFont('Consolas', 12))
        self.lbl_emotion = QLabel('Эмоция: —')
        self.lbl_emotion.setFont(QFont('Consolas', 12))
        self.lbl_conf = QLabel('Уверенность: —')
        self.lbl_conf.setFont(QFont('Consolas', 12))
        self.lbl_threshold_val = QLabel('Порог: 0.3')
        self.lbl_threshold_val.setFont(QFont('Consolas', 12))
        info.addWidget(self.lbl_fps)
        info.addWidget(self.lbl_emotion)
        info.addWidget(self.lbl_conf)
        info.addWidget(self.lbl_threshold_val)
        root.addLayout(info)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel('Порог уверенности:'))
        from PyQt5.QtWidgets import QSlider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.lbl_threshold_val.setText(f'Порог: {v / 100:.2f}'))
        config_row.addWidget(self.threshold_slider)
        self.conf_threshold = 0.30
        self.threshold_slider.valueChanged.connect(
            lambda v: setattr(self, 'conf_threshold', v / 100))
        root.addLayout(config_row)

    def _load_emotion_model(self):
        arch_map = {'MobileNetV2': 'mobilenet', 'ResNet18': 'resnet18'}
        self.current_arch = arch_map.get(self.arch_combo.currentText(), 'mobilenet')
        default = os.path.join(BASE, 'models', 'best_model.pt')
        path, _ = QFileDialog.getOpenFileName(self, 'Выберите веса модели', default, '*.pt')
        if not path:
            return
        try:
            self.emotion_model = build_emotion_model(self.current_arch, path, self.device)
            self.statusBar().showMessage(f'Модель загружена: {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', str(e))

    def _toggle_camera(self):
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.btn_start.setText('▶ Старт')
            self.statusBar().showMessage('Камера остановлена.')
            return
        idx = self.cam_combo.currentIndex()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            QMessageBox.warning(self, 'Камера', f'Не удалось открыть камеру {idx}')
            return
        res_text = self.res_combo.currentText()
        try:
            rw, rh = map(int, res_text.split('x'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        except ValueError:
            pass
        self.timer.start(30)
        self.btn_start.setText('⏹ Стоп')
        self.statusBar().showMessage(f'Камера {idx} запущена.')

    def _process_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        t0 = time.time()
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self._process_emotions(frame)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg = np.mean(self.fps_history)
        self.lbl_fps.setText(f'FPS: {avg:.1f}')

        overlay_text = f'FPS: {avg:.1f}'
        cv2.putText(frame, overlay_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self._display(frame, self.video_label)

    def _process_emotions(self, frame):
        if self.emotion_model is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        best_label, best_conf = 'neutral', 0.0
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if len(face.shape) == 3 else cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.emotion_model(tensor)
                probs = torch.softmax(out, dim=1)[0]
            idx = probs.argmax().item()
            label = EMOTION_CLASSES[idx]
            conf = probs[idx].item()
            color = EMOTION_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            ru = EMOTION_RU.get(label, label)
            if conf >= self.conf_threshold:
                text = f'{ru} {conf:.0%}'
                cv2.putText(frame, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)
            if conf > best_conf:
                best_label, best_conf = label, conf
        self.lbl_emotion.setText(f'Эмоция: {EMOTION_RU.get(best_label, best_label)}')
        self.lbl_conf.setText(f'Уверенность: {best_conf:.0%}')
        return frame

    def _display(self, frame, label_widget):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        label_widget.setPixmap(QPixmap.fromImage(qimg).scaled(
            label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _blind_test(self):
        """Слепое тестирование: выбрать папку с подпапками (angry, happy, ...)
        и получить accuracy + confusion matrix. Или одно изображение."""
        msg = QMessageBox(self)
        msg.setWindowTitle('Blind Test')
        msg.setText('Что тестировать?')
        btn_folder = msg.addButton('Папку с классами', QMessageBox.ActionRole)
        btn_image = msg.addButton('Одно изображение', QMessageBox.ActionRole)
        msg.addButton('Отмена', QMessageBox.RejectRole)
        msg.exec_()

        if msg.clickedButton() == btn_image:
            self._blind_test_single()
        elif msg.clickedButton() == btn_folder:
            self._blind_test_folder()

    def _blind_test_single(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение',
                                               '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        if self.emotion_model is None:
            QMessageBox.warning(self, 'Внимание', 'Сначала загрузите модель.')
            return
        frame = _safe_imread(path)
        if frame is None:
            QMessageBox.warning(self, 'Ошибка', f'Не удалось прочитать изображение:\n{path}')
            return
        frame = self._process_emotions(frame)
        self._display(frame, self.video_label)

    def _blind_test_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Выберите папку test/ с подпапками эмоций')
        if not folder:
            return
        if self.emotion_model is None:
            QMessageBox.warning(self, 'Внимание', 'Сначала загрузите модель.')
            return

        from utils.dataset import EMOTION_CLASSES, get_test_transforms, FER2013FolderDataset
        from utils.metrics import compute_and_save_metrics

        self.statusBar().showMessage('Blind test: идёт оценка...')
        QApplication.processEvents()

        ds = FER2013FolderDataset(folder, transform=get_test_transforms(self.img_size))
        if len(ds) == 0:
            QMessageBox.warning(self, 'Ошибка', 'В папке нет изображений.\n'
                                'Структура: test/angry/*.jpg, test/happy/*.jpg ...')
            return

        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        y_true, y_pred, y_probs_list = [], [], []
        correct, total = 0, 0

        self.emotion_model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.emotion_model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                correct += preds.eq(labels.to(self.device)).sum().item()
                total += labels.size(0)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs_list.append(probs.cpu().numpy())

        acc = correct / total if total > 0 else 0
        y_probs_all = np.concatenate(y_probs_list, axis=0)

        results_dir = os.path.join(BASE, 'results')
        os.makedirs(results_dir, exist_ok=True)
        compute_and_save_metrics(y_true, y_pred, y_probs_all, EMOTION_CLASSES, results_dir)

        QMessageBox.information(self, 'Blind Test — Результат',
                                f'Протестировано: {total} изображений\n'
                                f'Accuracy: {acc:.2%}\n\n'
                                f'Confusion matrix и ROC сохранены в:\n{results_dir}/')
        self.statusBar().showMessage(f'Blind test done: accuracy={acc:.2%} ({total} imgs)')

    def _show_results_image(self, filename, title):
        """Open a resizable dialog showing an image from results/ directory with zoom controls."""
        path = os.path.join(BASE, 'results', filename)

        if not os.path.isfile(path) and filename == 'training_curves.png':
            history_path = os.path.join(BASE, 'results', 'training_history.json')
            if os.path.isfile(history_path):
                try:
                    import json
                    from utils.metrics import plot_training_curves
                    with open(history_path, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    plot_training_curves(history, os.path.join(BASE, 'results'))
                    self.statusBar().showMessage('training_curves.png сгенерирован из истории обучения.')
                except Exception as e:
                    QMessageBox.warning(self, 'Ошибка', f'Не удалось построить кривые:\n{e}')
                    return

        if not os.path.isfile(path):
            QMessageBox.warning(self, 'Файл не найден',
                                f'Файл не найден:\n{path}\n\n'
                                'Сначала запустите обучение (train.py) или Blind Test.')
            return

        dlg = ImageViewerDialog(self, path, title)
        dlg.exec_()

    def _show_metrics_json(self):
        """Show metrics.json content in a resizable dialog with copyable text."""
        path = os.path.join(BASE, 'results', 'metrics.json')
        if not os.path.isfile(path):
            QMessageBox.warning(self, 'Файл не найден',
                                f'metrics.json не найден:\n{path}\n\n'
                                'Сначала запустите обучение или Blind Test.')
            return
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось прочитать metrics.json:\n{e}')
            return

        dlg = QDialog(self)
        dlg.setWindowTitle('metrics.json')
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)
        try:
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(data)
        text.setPlainText(pretty)
        font = text.font()
        font.setFamily('Consolas' if os.name == 'nt' else 'monospace')
        text.setFont(font)

        layout.addWidget(text)

        btn_close = QPushButton('Закрыть')
        btn_close.clicked.connect(dlg.close)
        layout.addWidget(btn_close)

        dlg.exec_()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
