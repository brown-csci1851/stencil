"""Data loader for Homework 6: GNNs"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data


class HW6DataLoader:
    def __init__(self):
        """Initialize data loader with cache directory for datasets."""
        self.homework_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.homework_dir / "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def make_node_features(self, num_nodes, embedding_dim, feature_type, node_descriptions=None):
        # TODO: Return node features for one of the following feature types:
        # - "random": random dense features of shape [num_nodes, embedding_dim]
        # - "one-hot": identity-matrix features of shape [num_nodes, num_nodes]
        # - "pretrained": optional extension using node_descriptions to build
        #   description-based features for each drug or disease
        #
        # For the optional extension, reasonable starting points include:
        # - TF-IDF vectors built from the text descriptions
        # - sentence/document embeddings from a pretrained text model
        # - any fixed-length text embedding pipeline that maps each description
        #   to one vector per node
        pass

    def get_drug_disease_data_split(
        self,
        train_csv: str,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        embedding_dim: int = 32,
        feature_type: str = "random",
        val_ratio: float = 0.2,
        seed: int = 42,
        drug_col: str = "Drug_ID",
        disease_col: str = "Disease_ID",
        rel_col: str = "Y",
        drug_name_col: str = "Drug_Name",
        disease_name_col: str = "Disease_Name",
        drug_desc_col: str = "Drug_Description",
        disease_desc_col: str = "Disease_Description",
    ) -> Data:
        # TODO: Read train/test CSV files and, if provided, val.csv.
        # TODO: Split train.csv into train/val edges
        # (preferably stratified by rel_col) using val_ratio and seed.
        # TODO: Build node2id and rel2id mappings.
        # Use distinct internal keys for drugs and diseases so overlapping raw
        # IDs do not collapse into one node.
        # TODO: Write a helper that converts a DataFrame into edge_index and edge_type.
        # Use drug_col, disease_col, and rel_col for indexing.
        # TODO: Create full-graph edges plus train/val/test split tensors.
        # Keep the full graph in data.edge_index / data.edge_type for analysis,
        # but attach data.message_passing_edge_index /
        # data.message_passing_edge_type using only the training edges, and make
        # that message-passing graph bidirectional so both drugs and diseases can
        # exchange information without leaking validation/test edges.
        # TODO: Call self.make_node_features(...) to create random, one-hot, or
        # optional description-based pretrained features. If you implement the
        # optional extension, pass the ordered node descriptions into that helper.
        # TODO: Store node/relation metadata on the Data object so embedding plots
        # can be mapped back to biomedical entities and interaction labels.
        # In particular, preserve namespaced node IDs, raw entity IDs, display
        # names, descriptions, and node types.
        pass

    def get_drug_drug_data_split(
        self,
        train_csv: str,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        embedding_dim: int = 32,
        feature_type: str = "random",
        val_ratio: float = 0.2,
        seed: int = 42,
        drug1_col: str = "Drug1_ID",
        drug2_col: str = "Drug2_ID",
        rel_col: str = "Y",
        drug1_name_col: str = "Drug1_Name",
        drug2_name_col: str = "Drug2_Name",
        drug1_desc_col: str = "Drug1_Description",
        drug2_desc_col: str = "Drug2_Description",
    ) -> Data:
        # TODO: Read train/test CSV files for the drug-drug dataset.
        # TODO: Split train.csv into train/val edges
        # (preferably stratified by rel_col) using val_ratio and seed.
        # TODO: Build drug2id and rel2id mappings.
        # TODO: Write a helper that converts a DataFrame into edge_index and edge_type
        # using drug1_col, drug2_col, and rel_col.
        # TODO: Create full-graph edges plus train/val/test split tensors.
        # Keep the full graph in data.edge_index / data.edge_type for analysis,
        # but attach data.message_passing_edge_index /
        # data.message_passing_edge_type using only the training edges, and make
        # that message-passing graph bidirectional so both endpoints can exchange
        # information without leaking validation/test edges.
        # TODO: Call self.make_node_features(...) to create random, one-hot, or
        # optional description-based pretrained features. If you implement the
        # optional extension, pass the ordered node descriptions into that helper.
        # TODO: Store node/relation metadata on the Data object so embedding plots
        # and model comparisons can be interpreted later, including node IDs, raw
        # entity IDs, display names, descriptions, and node types.
        pass
