# Import necessary libraries
!pip install -q torchsummary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as opt
from torchsummary import summary

# Configuration class
class cfg:
    batch_size = 32
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((28, 28))
        ]
    )
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4

# Load datasets
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

# Custom Dataset class for training data
class DigitDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype("float32").reshape(28, 28) / 255.0
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Custom Dataset class for test data
class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.data.iloc[idx, :].values.astype("float32").reshape(28, 28)
        if self.transform:
            img = self.transform(img)
        return img

# Prepare datasets and dataloaders
x, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
train = DigitDataset(x, y, transform=cfg.transform)
test = TestDataset(test, transform=cfg.transform)

train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
test = DataLoader(test, batch_size=cfg.batch_size, shuffle=False)

# Define the CNN model
class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate and summarize the model
model = Model().to(cfg.device)
summary(model, (1, 28, 28))

# Define optimizer, criterion, and scheduler
optimizer = opt.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
for epoch in range(cfg.epochs):
    running_loss = 0.0
    for img, label in train_loader:
        img = img.to(cfg.device)
        label = label.to(cfg.device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader)}")

# Evaluate model accuracy on training data
correct = 0
total = 0
with torch.no_grad():
    for img, label in train_loader:
        img = img.to(cfg.device)
        label = label.to(cfg.device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

# Generate predictions for test dataset
prediction = []
with torch.no_grad():
    for img in test:
        img = img.to(cfg.device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        prediction.extend(predicted.cpu().numpy())

# Create submission file
submission = pd.DataFrame({'ImageId': list(range(1, len(prediction) + 1)), 'Label': prediction})
submission.to_csv('submission.csv', index=False)
