"""Data loader for Homework 5: Autoencoders"""

import os
from pathlib import Path
import numpy as np
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader


class SingleCellDataset(Dataset):
    def __init__(self, counts: np.ndarray, labels: np.ndarray):
        """
        counts: np.ndarray of shape [num_cells, num_genes]
        labels: np.ndarray of shape [num_cells]
        """
        self.counts = torch.tensor(counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        return self.counts[idx], self.labels[idx]


class HW5DataLoader:
    def __init__(self):
        """Initialize data loader with cache directory for datasets"""
        self.homework_dir = Path(__file__).resolve().parent.parent 
        self.data_dir = self.homework_dir / "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def get_simple_data(self, batch_size: int = 128, shuffle: bool = True, data_path: Optional[str] = None) -> DataLoader:
        """
        Loads the default counts.npy and labels.txt from the data directory
        and returns a PyTorch DataLoader.

        Returns:
            DataLoader yielding (input_tensor, label_tensor)
        """
        # Resolve paths
        counts_path = data_path or self.data_dir / "counts.npy"
        labels_path = data_path or self.data_dir / "labels.txt"

        # Load data
        counts = np.load(counts_path)
        labels = np.loadtxt(labels_path)

        dataset = SingleCellDataset(counts, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def load_custom_data(self, counts_file: str, labels_file: str, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
        """
        Loads a custom counts + labels file and returns a DataLoader.

        Args:
            counts_file: Path to counts.npy file
            labels_file: Path to labels.txt file

        Returns:
            DataLoader
        """
        counts = np.load(counts_file)
        labels = np.loadtxt(labels_file)

        dataset = SingleCellDataset(counts, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
