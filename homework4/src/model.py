import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # TODO: define a small MLP (e.g., Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear)
        # TODO: keep output size = 1 for binary classification
        # TODO: try different hidden sizes and dropout rates
        pass

    def forward(self, x):
        # TODO: flatten input to (batch, input_dim) and pass through the MLP
        pass

class CNNClassifier(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # TODO: define a few conv blocks (Conv2d -> ReLU -> MaxPool)
        # TODO: optionally add BatchNorm2d and Dropout
        # TODO: compute the flattened size and add a small classifier head (Linear layers)
        pass

    def forward(self, x):
        # TODO: run conv blocks, flatten, then run the classifier head
        pass
