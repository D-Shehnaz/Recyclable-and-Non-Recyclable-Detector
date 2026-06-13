import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cnn import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data augmentation + preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Dataset
dataset = datasets.ImageFolder(
    root="dataset_cnn/images/train",
    transform=transform
)

# Check classes
print("\nClasses Found:")
print(dataset.classes)

print("\nClass Mapping:")
print(dataset.class_to_idx)

num_classes = len(dataset.classes)

print("\nNumber of Classes:", num_classes)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# Model
model = get_model(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

# Training Loop
for epoch in range(epochs):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_acc:.2f}%"
    )

# Save model
torch.save(model.state_dict(), "cnn_model/model.pth")

print("\nModel saved to cnn_model/model.pth")