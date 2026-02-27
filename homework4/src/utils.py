import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# TODO: implement confusion matrix plotting
def plot_confusion_matrix(cm, class_names=["Normal", "Pneumonia"]):
    pass

# TODO: implement activation visualization
def visualize_activations(model, input_tensor, layer_name='conv1'):
    pass

def plot_class_distribution(loader, title="Class Distribution"):
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    count = Counter(labels)
    plt.figure(figsize=(5, 3))
    sns.barplot(x=list(count.keys()), y=list(count.values()))
    plt.xticks([0, 1], ["Normal", "Pneumonia"])
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()

def plot_training_curves(history):
    if not history:
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_accuracy = history.get("val_accuracy", [])
    val_auroc = history.get("val_auroc", [])

    max_len = max(
        len(train_loss),
        len(val_loss),
        len(val_accuracy),
        len(val_auroc),
        0,
    )
    if max_len == 0:
        return

    epochs = range(1, max_len + 1)
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(10, 4))

    if train_loss:
        ax_loss.plot(epochs[:len(train_loss)], train_loss, label="Train Loss")
    if val_loss:
        ax_loss.plot(epochs[:len(val_loss)], val_loss, label="Val Loss")
    ax_loss.set_title("Loss vs Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    if val_accuracy:
        ax_metric.plot(epochs[:len(val_accuracy)], val_accuracy, label="Val Accuracy")
    if val_auroc:
        ax_metric.plot(epochs[:len(val_auroc)], val_auroc, label="Val AUROC")
    ax_metric.set_title("Metric vs Epoch")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Metric")
    ax_metric.legend()

    fig.tight_layout()
    plt.show()
    
def visualize_samples(loader, grayscale=True):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img = images[i][0] if grayscale else images[i].permute(1, 2, 0)
        axes[i].imshow(img, cmap='gray' if grayscale else None)
        axes[i].set_title(f"Label: {int(labels[i])}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
