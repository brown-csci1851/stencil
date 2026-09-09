import torch
import torch.nn as nn


class FCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 13,
        hidden_dims=(512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        # TODO: define a small fully connected network.
        # Suggested structure:
        #   Linear(input_dim -> hidden_dims[0])
        #   ReLU
        #   Dropout
        #   Linear(hidden_dims[0] -> hidden_dims[1])
        #   ReLU
        #   Dropout
        #   Linear(hidden_dims[1] -> num_classes)
        #
        # The output should be raw logits with shape (batch_size, num_classes).
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: flatten x to shape (batch_size, input_dim), then pass it
        # through your fully connected network.
        pass


class CNNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 13,
        dropout: float = 0.3,
    ):
        super().__init__()
        # TODO: define a 1D CNN for RNA sequences.
        #
        # Input shape from the loader:
        #   (batch_size, 5, max_length)
        #
        # Suggested structure:
        #   Conv1d -> ReLU -> MaxPool1d
        #   Conv1d -> ReLU -> MaxPool1d
        #   optional Conv1d / BatchNorm1d / Dropout
        #   global pooling or flattening
        #   Linear(... -> num_classes)
        #
        # The output should be raw logits with shape (batch_size, num_classes).
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: run x through your convolutional layers and classifier head.
        pass
