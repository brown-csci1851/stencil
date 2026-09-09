from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


MODEL_DIR = "results"

# Multiclass classification uses raw logits and integer class labels.
criterion = nn.CrossEntropyLoss()


def evaluate_model(model, dataloader, device):
    # TODO: set the model to evaluation mode.
    # TODO: disable gradient computation with torch.no_grad().
    # TODO: iterate over dataloader batches and move x/y to device.
    # TODO: run the model to get logits with shape (batch_size, num_classes).
    # TODO: compute validation/test loss using criterion(logits, y).
    # TODO: convert logits to predicted class ids with argmax(dim=1).
    # TODO: collect predictions and true labels on the CPU.
    # TODO: compute accuracy_score, macro f1_score, and confusion_matrix.
    # TODO: return a dictionary with keys:
    #   "loss", "accuracy", "macro_f1", "confusion_matrix"
    pass


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    lr=1e-3,
    save_path=f"{MODEL_DIR}/best_model.pt",
    model_name="Model",
):
    # TODO: move model to device.
    # TODO: create an Adam optimizer.
    # TODO: create a history dictionary with:
    #   train_loss, val_loss, val_accuracy, val_macro_f1
    # TODO: create the directory for save_path if it does not exist.
    #
    # For each epoch:
    # TODO: set model.train().
    # TODO: loop through train_loader.
    # TODO: move x/y to device.
    # TODO: zero gradients, run forward pass, compute CrossEntropyLoss.
    # TODO: backpropagate and step the optimizer.
    # TODO: compute average train loss.
    # TODO: evaluate on val_loader with evaluate_model().
    # TODO: append losses and metrics to history.
    # TODO: print an epoch summary.
    # TODO: save the model state_dict when validation macro F1 improves.
    #
    # TODO: return history.
    pass


def train_cnn_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    lr=1e-3,
    save_path=f"{MODEL_DIR}/cnn_best_model.pt",
):
    # TODO: call train_model with model_name="CNN".
    pass


def train_fcn_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    lr=1e-3,
    save_path=f"{MODEL_DIR}/fcn_best_model.pt",
):
    # TODO: call train_model with model_name="FCN".
    pass


def eval_model(model, test_loader, device, model_path):
    # TODO: load the saved model state_dict from model_path.
    # TODO: move the model to device.
    # TODO: call evaluate_model on test_loader.
    # TODO: print test accuracy and macro F1.
    # TODO: return the metrics dictionary.
    pass


def eval_cnn_model(
    model,
    test_loader,
    device,
    model_path=f"{MODEL_DIR}/cnn_best_model.pt",
):
    # TODO: call eval_model for the CNN.
    pass


def eval_fcn_model(
    model,
    test_loader,
    device,
    model_path=f"{MODEL_DIR}/fcn_best_model.pt",
):
    # TODO: call eval_model for the FCN.
    pass
