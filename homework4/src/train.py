import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

MODEL_DIR = "results"

criterion = nn.BCEWithLogitsLoss()

def evaluate_model(model, dataloader, device):
    # TODO: set model.eval(), disable grad, iterate batches, move x/y to device
    # TODO: run forward to logits, apply sigmoid to get probabilities
    # TODO: collect probs and targets on CPU lists for metric computation
    # TODO: threshold probs at 0.5 for class predictions
    # TODO: compute accuracy_score, roc_auc_score, and confusion_matrix
    # TODO: return a dict with keys: "accuracy", "auroc", "confusion_matrix"
    pass

def train_cnn_model(model, train_loader, val_loader, epochs, device, lr=1e-3, save_path=f"{MODEL_DIR}/cnn_best_model.pt"):
    # TODO: move model to device; create Adam optimizer with lr
    # TODO: initialize best_val_auroc and history dict with lists:
    #       train_loss, val_loss, val_accuracy, val_auroc
    # TODO: for each epoch:
    #       - set model.train()
    #       - loop over train_loader: forward, compute BCEWithLogitsLoss,
    #         backward, optimizer.step(), accumulate running_loss
    #       - compute avg train loss
    #       - run evaluate_model on val_loader for accuracy/auroc
    #       - compute val loss with no_grad (separate pass over val_loader)
    #       - append metrics/losses to history
    #       - print epoch summary
    #       - if val AUROC improves, save model state_dict
    # TODO: print best AUROC and return history
    pass

def eval_cnn_model(model, test_loader, device, model_path=f"{MODEL_DIR}/cnn_best_model.pt"):
    # TODO: load state_dict from model_path, move to device, set eval
    # TODO: call evaluate_model on test_loader
    # TODO: print accuracy and AUROC, return metrics dict
    pass

def train_fcn_model(model, train_loader, val_loader, epochs, device, lr=1e-3, save_path=f"{MODEL_DIR}/fcn_best_model.pt"):
    # TODO: same structure as train_cnn_model, but for FCN inputs (flattened)
    # TODO: track history: train_loss, val_loss, val_accuracy, val_auroc
    # TODO: return history for plotting
    pass

def eval_fcn_model(model, test_loader, device, model_path=f"{MODEL_DIR}/fcn_best_model.pt"):
    # TODO: load FCN state_dict, move to device, set eval
    # TODO: call evaluate_model, print accuracy/AUROC, return metrics
    pass
