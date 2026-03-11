"""
FER-2013 Dataset loader for PyTorch.
Supports loading from folder structure: train/<class_name>/*.jpg
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)


def get_train_transforms(img_size=48):
    """Аугментации для обучения (критерий Оптимизация — аугментация данных)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms(img_size=48):
    """Преобразования для валидации/теста (без аугментаций)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class FER2013FolderDataset(Dataset):
    """
    Загрузка FER-2013 из структуры папок:
        root/
            angry/  img1.jpg img2.jpg ...
            disgust/ ...
            ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(EMOTION_CLASSES)}

        for cls_name in EMOTION_CLASSES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(train_dir, test_dir, batch_size=64, img_size=48, num_workers=2):
    """Возвращает train_loader и test_loader."""
    train_ds = FER2013FolderDataset(train_dir, transform=get_train_transforms(img_size))
    test_ds = FER2013FolderDataset(test_dir, transform=get_test_transforms(img_size))

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin)
    return train_loader, test_loader
