"""Contains code for Project 2 -- Garbage Recycling, Improved"""

import os
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet Mean and Std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PICTURE_SIZE = 224
CATEGORIES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 100
file_name, _ = os.path.splitext(os.path.basename(__file__))

# Preprocessor
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(PICTURE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(PICTURE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
)

# Dataset import
dataset_train = datasets.ImageFolder("./splited_data/train", transform=train_transform)
dataset_val = datasets.ImageFolder("./splited_data/val", transform=val_test_transform)
dataset_test = datasets.ImageFolder("./splited_data/test", transform=val_test_transform)


train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class TrashRecycleCNN(nn.Module):
    """Convolution Neural Network for Trash Recycling"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        # Non-Linearity: ReLU
        # Max-Pool
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        # Non-Linearity: ReLU
        # Max-Pool
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        # Non-Linearity: ReLU
        # Adaptive Avg Pool

        # View
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        # Non-Linearity: ReLU
        self.do = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, CATEGORIES) # Output

    def forward(self, x):
        """Implemented forward method"""
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = F.adaptive_avg_pool2d(x, (7, 7))

        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrashRecycleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# History Dictionary
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_acc = 0.0

for epoch in range(NUM_EPOCHS):

    model.train() # Start Training
    total_loss = 0
    total = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

    history['train_loss'].append(total_loss / total)
    history['train_acc'].append(100 * correct / total)

    model.eval() # End Training, and then Validate.

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_running_loss += loss.item()
        history['val_loss'].append(val_running_loss / val_total)
        history['val_acc'].append(100 * val_correct / val_total)

    # Serialize the history data
    with open(f"result/{file_name}.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%") # Test Accuracy: 86.99%
