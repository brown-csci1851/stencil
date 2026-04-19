import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv

# TODO: Complete this class for R-GCN
class RGCNLinkPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_layers=2):
        super().__init__()
        # TODO: Define RGCNConv layers 
        # The first layer should map in_dim -> out_dim and the remaining
        # layers should map out_dim -> out_dim.
        pass

    def forward(self, x, edge_index, edge_type):
        # TODO: Pass through each RGCNConv layer and apply non-linearity
        pass

    def predict(self, node_embeddings, edge_index):
        # TODO: Implement dot-product-based link prediction
        pass


# TODO: Implement a simple GCN link predictor
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        # TODO: Define GCNConv layers using nn.ModuleList.
        # The first layer should map in_dim -> out_dim and the remaining
        # layers should map out_dim -> out_dim.
        pass

    def forward(self, x, edge_index, edge_type=None):
        # TODO: Apply GCNConv layers.
        # edge_type is unused for GCN, but keeping it in the signature lets
        # the model share the same training loop as the R-GCN version.
        pass

    def predict(self, node_embeddings, edge_index):
        # TODO: Implement dot-product-based link prediction
        pass
