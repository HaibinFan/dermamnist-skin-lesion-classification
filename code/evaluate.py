import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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
# Test transform
# =========================
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =========================
# Dataset
# =========================
test_dataset = DermaMNIST(
    split="test",
    transform=test_transform,
    download=True,
    size=224,
    root=data_root
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

num_classes = len(test_dataset.info["label"])
class_names = [test_dataset.info["label"][str(i)] for i in range(num_classes)]

print("Number of classes:", num_classes)
print("Class names:", class_names)

# =========================
# Evaluation function
# =========================
def evaluate_model(model, model_name, weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n===== {model_name} Test Results =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)

    # save metrics
    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1)
    }

    metrics_path = os.path.join(save_root, f"{model_name}_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    report_path = os.path.join(save_root, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join(save_root, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    # Evaluate SimpleCNN
    simple_cnn = SimpleCNN(num_classes=num_classes)
    evaluate_model(
        model=simple_cnn,
        model_name="simple_cnn",
        weight_path=os.path.join(save_root, "simple_cnn_best.pth")
    )

    # Evaluate ResNet18
    resnet18 = get_resnet18(num_classes=num_classes)
    evaluate_model(
        model=resnet18,
        model_name="resnet18",
        weight_path=os.path.join(save_root, "resnet18_best.pth")
    )