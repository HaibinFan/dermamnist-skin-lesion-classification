import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST

from models import SimpleCNN, get_resnet18

# =========================
# Paths
# =========================
data_root = "/blue/bme6938/haibinfan/project2/data"
save_root = "/blue/bme6938/haibinfan/project2/results"
os.makedirs(save_root, exist_ok=True)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =========================
# Datasets
# =========================
train_dataset = DermaMNIST(
    split="train",
    transform=train_transform,
    download=True,
    size=224,
    root=data_root
)

val_dataset = DermaMNIST(
    split="val",
    transform=test_transform,
    download=True,
    size=224,
    root=data_root
)

test_dataset = DermaMNIST(
    split="test",
    transform=test_transform,
    download=True,
    size=224,
    root=data_root
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

num_classes = len(train_dataset.info["label"])
print("Number of classes:", num_classes)

# =========================
# Train / Eval functions
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(model, model_name, epochs, lr, weight_decay=1e-4, patience=5):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{model_name} Epoch [{epoch+1}/{epochs}] "
            f"LR: {current_lr:.6f} "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_root, f"{model_name}_best.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered for {model_name} at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)

    history_path = os.path.join(save_root, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Best validation accuracy for {model_name}: {best_val_acc:.4f}")
    print(f"Best epoch for {model_name}: {best_epoch}")

    return model, history, best_val_acc


if __name__ == "__main__":
    # Simple CNN
    cnn_model = SimpleCNN(num_classes=num_classes)
    run_training(
        model=cnn_model,
        model_name="simple_cnn",
        epochs=15,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5
    )

    # ResNet18
    resnet_model = get_resnet18(num_classes=num_classes)
    run_training(
        model=resnet_model,
        model_name="resnet18",
        epochs=15,
        lr=1e-4,
        weight_decay=1e-4,
        patience=5
    )