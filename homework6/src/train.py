import torch
from utils import negative_sampling

def train(model, optimizer, data, edge_index, edge_type, epoch=None, log_path=None):
    model.train()
    optimizer.zero_grad()

    # TODO: Compute node embeddings using the train-only message-passing graph
    # stored in data.message_passing_edge_index (and
    # data.message_passing_edge_type for R-GCN).
    # TODO: Compute positive scores for the split edges passed in through
    # edge_index using model.predict.
    # TODO: Generate negative samples with negative_sampling(
    # data, ..., excluded_edge_index=data.train_pos_edge_index
    # ) and compute scores.

    # TODO: Calculate loss (binary cross-entropy)
    # TODO: Backpropagation and optimizer step
    # TODO: Log training

    return 0.0  # TODO: return actual loss

@torch.no_grad()
def evaluate(model, data, edge_index, edge_type, threshold=0.5):
    model.eval()

    # TODO: Compute embeddings from the train-only message-passing graph stored
    # in data.message_passing_edge_index.
    # TODO: Score the positive split edges passed in through edge_index and a
    # matching set of negative edges from negative_sampling(
    # data, ..., excluded_edge_index=data.edge_index
    # ).
    # TODO: Convert scores to binary predictions using the threshold
    # TODO: Calculate accuracy

    return { # TODO: replace with actual metrics
        "accuracy": 0.0,
        "f1": 0.0
        }
