#!/usr/bin/env python

import os
import random
from collections import Counter

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

BATCH_SIZE = 16
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, split_ratio=0.9):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.images = []
        self.labels = []
        self.classes = ["brightness", "invert"]

        temp_images = []
        temp_labels = []

        for idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                temp_images.append(os.path.join(cls_folder, img_name))
                temp_labels.append(idx)

        counter = Counter(temp_labels)
        max_count = max(counter.values())
        for cls_idx in counter.keys():
            cls_indices = [i for i, x in enumerate(temp_labels) if x == cls_idx]
            while counter[cls_idx] < max_count:
                index = random.choice(cls_indices)
                temp_images.append(temp_images[index])
                temp_labels.append(temp_labels[index])
                counter[cls_idx] += 1

        dataset_size = len(temp_images)
        split = int(np.floor(split_ratio * dataset_size))
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        if self.train:
            indices = indices[:split]
        else:
            indices = indices[split:]

        self.images = [temp_images[i] for i in indices]
        self.labels = [temp_labels[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def init_model():
    model = timm.create_model("efficientnet_b0", pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 1), nn.Sigmoid())
    model = model.to(device)
    return model


def train_model(model, train_loader, test_loader):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_items = 0
        tqdm_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", total=len(train_loader))

        for inputs, labels in tqdm_bar:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += correct_count(outputs, labels)
            total_items += inputs.size(0)

            avg_loss = train_loss / total_items
            tqdm_bar.set_postfix(loss=avg_loss)

        lr_scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += correct_count(outputs, labels)

        train_loss /= total_items
        train_accuracy = train_correct / total_items
        val_loss /= len(test_loader.dataset)
        val_accuracy = val_correct / len(test_loader.dataset)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Epoch {epoch+1}: Model saved with validation accuracy: {val_accuracy:.2f}")

        print(
            f"Epoch {epoch+1}/{EPOCHS}. "
            f"Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.2f}. "
            f"Val Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.2f}"
        )

    torch.save(model.state_dict(), "model.pth")


def correct_count(outputs, labels):
    preds = (outputs > 0.5).float()
    return (preds == labels).float().sum()


if __name__ == "__main__":
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(160),
            v2.RandomCrop(128),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(128),
            v2.CenterCrop(128),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    train_dataset = CustomDataset(root_dir="dataset/", transform=train_transform, train=True)
    test_dataset = CustomDataset(root_dir="dataset/", transform=test_transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = init_model()
    train_model(model, train_loader, test_loader)
