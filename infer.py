"""
infer.py — Задание 1: Распознавание эмоций в видеопотоке.

Использование:
    python infer.py                  — камера 0
    python infer.py --source 1       — камера 1
    python infer.py --image photo.jpg — одно изображение
    python infer.py --model resnet18 --weights models/best_model.pt
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import EMOTION_CLASSES, NUM_CLASSES

EMOTION_COLORS = {
    'angry':    (0, 0, 255),
    'disgust':  (0, 128, 0),
    'fear':     (128, 0, 128),
    'happy':    (0, 255, 255),
    'neutral':  (200, 200, 200),
    'sad':      (255, 0, 0),
    'surprise': (0, 165, 255),
}

EMOTION_RU = {
    'angry': 'Злость', 'disgust': 'Отвращение', 'fear': 'Страх',
    'happy': 'Радость', 'neutral': 'Нейтрально', 'sad': 'Грусть',
    'surprise': 'Удивление',
}


def build_model(arch, num_classes, weights_path, device):
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    else:
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, num_classes))
    if weights_path and os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f'[infer] Loaded weights: {weights_path}')
    else:
        print(f'[infer] WARNING: weights not found at {weights_path}')
    model.to(device).eval()
    return model


def get_transform(img_size=48):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_face_cascade():
    import shutil
    import tempfile
    cascade_name = 'haarcascade_frontalface_default.xml'
    cascade_path = cv2.data.haarcascades + cascade_name
    cascade = cv2.CascadeClassifier(cascade_path)
    if not cascade.empty():
        return cascade
    tmp_path = os.path.join(tempfile.gettempdir(), cascade_name)
    if not os.path.isfile(tmp_path):
        shutil.copy2(cascade_path, tmp_path)
    cascade = cv2.CascadeClassifier(tmp_path)
    if cascade.empty():
        print('[infer] ERROR: Could not load Haar cascade')
        sys.exit(1)
    print(f'[infer] Haar cascade loaded from temp: {tmp_path}')
    return cascade


def predict_emotion(model, face_img, transform, device):
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 1:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    idx = probs.argmax().item()
    return EMOTION_CLASSES[idx], probs[idx].item(), probs.cpu().numpy()


def draw_result(frame, x, y, w, h, label, conf):
    color = EMOTION_COLORS.get(label, (255, 255, 255))
    ru_label = EMOTION_RU.get(label, label)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f'{ru_label} {conf:.0%}'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def run_camera(model, device, transform, cascade, source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[infer] Cannot open camera {source}')
        return
    print('[infer] Press Q to quit')
    fps_list = []
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label, conf, _ = predict_emotion(model, face, transform, device)
            draw_result(frame, x, y, w, h, label, conf)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        fps_list.append(fps)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if fps_list:
        print(f'[infer] Avg FPS: {np.mean(fps_list):.1f}')


def run_image(model, device, transform, cascade, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        try:
            with open(image_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            pass
    if frame is None:
        print(f'[infer] Cannot read image: {image_path}')
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        label, conf, probs = predict_emotion(model, face, transform, device)
        draw_result(frame, x, y, w, h, label, conf)
        print(f'  {EMOTION_RU.get(label, label)}: {conf:.2%}')
    cv2.imshow('Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Emotion inference')
    parser.add_argument('--source', type=int, default=0)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--model', type=str, default='mobilenet', choices=['resnet18', 'mobilenet'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=128)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    if args.weights is None:
        args.weights = os.path.join(base, 'models', 'best_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model, NUM_CLASSES, args.weights, device)
    transform = get_transform(args.img_size)
    cascade = load_face_cascade()

    if args.image:
        run_image(model, device, transform, cascade, args.image)
    else:
        run_camera(model, device, transform, cascade, args.source)


if __name__ == '__main__':
    main()
