from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_confusion_matrix(cm, class_names):
    # TODO: use seaborn.heatmap to visualize the confusion matrix.
    # TODO: label the x-axis as predicted labels and the y-axis as true labels.
    # TODO: use class_names for tick labels and rotate labels if needed.
    pass


def visualize_activations(model, input_tensor, layer_name="conv1", max_channels=8):
    # TODO: set model.eval().
    # TODO: register a forward hook on the requested Conv1d layer.
    # TODO: pass one input tensor through the model with a batch dimension.
    # TODO: remove the hook after the forward pass.
    # TODO: visualize the first max_channels activation channels as a heatmap.
    #
    # input_tensor should have shape:
    #   (5, max_length)
    pass


def plot_class_distribution(loader, class_names=None, title="Class Distribution"):
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())

    counts = Counter(labels)
    x_values = sorted(counts)
    x_labels = [class_names[idx] if class_names else str(idx) for idx in x_values]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=x_labels, y=[counts[idx] for idx in x_values])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()


def plot_sequence_length_distribution(dataset, title="Sequence Length Distribution"):
    plt.figure(figsize=(7, 4))
    sns.histplot(dataset.data["sequence_length"], bins=40)
    plt.title(title)
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    if not history:
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_accuracy = history.get("val_accuracy", [])
    val_macro_f1 = history.get("val_macro_f1", [])

    max_len = max(
        len(train_loss),
        len(val_loss),
        len(val_accuracy),
        len(val_macro_f1),
        0,
    )
    if max_len == 0:
        return

    epochs = range(1, max_len + 1)
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(10, 4))

    if train_loss:
        ax_loss.plot(epochs[: len(train_loss)], train_loss, label="Train Loss")
    if val_loss:
        ax_loss.plot(epochs[: len(val_loss)], val_loss, label="Val Loss")
    ax_loss.set_title("Loss vs Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    if val_accuracy:
        ax_metric.plot(epochs[: len(val_accuracy)], val_accuracy, label="Val Accuracy")
    if val_macro_f1:
        ax_metric.plot(epochs[: len(val_macro_f1)], val_macro_f1, label="Val Macro F1")
    ax_metric.set_title("Metric vs Epoch")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Metric")
    ax_metric.legend()

    fig.tight_layout()
    plt.show()


def visualize_samples(loader, class_names=None, num_samples=5):
    dataset = loader.dataset
    n = min(num_samples, len(dataset))

    for idx in range(n):
        row = dataset.data.iloc[idx]
        label_idx = int(row["label_idx"])
        label = class_names[label_idx] if class_names else row["label"]
        sequence = row["sequence"]
        print(f"Example {idx + 1}")
        print(f"Label: {label} ({label_idx})")
        print(f"Length: {row['sequence_length']}")
        print(f"Sequence preview: {sequence[:100]}")
        print()
