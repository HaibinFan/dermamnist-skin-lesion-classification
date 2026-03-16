import os
import json
import matplotlib.pyplot as plt

save_root = "/blue/bme6938/haibinfan/project2/results"

def plot_history(model_name):
    history_path = os.path.join(save_root, f"{model_name}_history.json")
    print(f"Reading: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    loss_path = os.path.join(save_root, f"{model_name}_loss_curve.png")
    acc_path = os.path.join(save_root, f"{model_name}_accuracy_curve.png")

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {loss_path}")
    print(f"Saved: {acc_path}")

if __name__ == "__main__":
    plot_history("simple_cnn")
    plot_history("resnet18")