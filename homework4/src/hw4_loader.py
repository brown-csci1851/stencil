"""
Data loader for Homework 4: RNA family classification with FCNs and 1D CNNs.

Expected data layout:
data/
  train.csv
  val.csv
  test.csv

Each CSV must contain:
sequence,sequence_length,label,label_idx
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class RNADataset(Dataset):
    """
    PyTorch dataset for fixed-length one-hot encoded RNA sequences.

    CNN examples are returned with shape (5, max_length), where channels are
    A, C, G, U, and N. FCN examples are flattened to shape (5 * max_length,).
    """

    channels = ("A", "C", "G", "U", "N")

    def __init__(
        self,
        csv_path: Path,
        max_length: int = 500,
        for_cnn: bool = True,
    ):
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV file not found: {csv_path}")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.csv_path = csv_path
        self.max_length = max_length
        self.for_cnn = for_cnn
        self.data = pd.read_csv(csv_path)

        required_columns = {"sequence", "sequence_length", "label", "label_idx"}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing_columns)}"
            )

        self.label_to_idx: Dict[str, int] = (
            self.data[["label", "label_idx"]]
            .drop_duplicates()
            .sort_values("label_idx")
            .set_index("label")["label_idx"]
            .to_dict()
        )
        self.idx_to_label: Dict[int, str] = {
            idx: label for label, idx in self.label_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        sequence = str(row["sequence"])
        label = int(row["label_idx"])

        x = self.one_hot_encode(sequence)
        if not self.for_cnn:
            x = x.reshape(-1)

        return x, torch.tensor(label, dtype=torch.long)

    def normalize_sequence(self, sequence: str) -> str:
        sequence = sequence.upper().replace("T", "U")
        valid_bases = set(self.channels[:-1])
        sequence = "".join(base if base in valid_bases else "N" for base in sequence)
        return sequence[: self.max_length].ljust(self.max_length, "N")

    def one_hot_encode(self, sequence: str) -> torch.Tensor:
        sequence = self.normalize_sequence(sequence)
        base_to_channel = {"A": 0, "C": 1, "G": 2, "U": 3}
        unknown_channel = 4

        x = torch.zeros((len(self.channels), self.max_length), dtype=torch.float32)
        for position, base in enumerate(sequence):
            channel = base_to_channel.get(base, unknown_channel)
            x[channel, position] = 1.0
        return x


class HW4DataLoader:
    def __init__(self, max_length: int = 500):
        self.homework_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.homework_dir / "data"
        self.dataset_dir = self.data_dir
        self.max_length = max_length

    def _check_dataset_exists(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Expected dataset folder not found: {self.dataset_dir}\n"
                "Make sure the RNA CSV files are placed in:\n"
                "  homework4/data/\n"
            )

    def get_rna_data(
        self,
        split: str = "train",
        for_cnn: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Load one RNA split.

        Args:
            split: One of train, val, or test.
            for_cnn: If True, return tensors with shape (batch, 5, max_length).
                If False, return flattened tensors with shape
                (batch, 5 * max_length).
            batch_size: Batch size for the DataLoader.
            num_workers: Number of worker processes for loading data.
        """
        self._check_dataset_exists()

        split_to_filename = {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
        }
        if split not in split_to_filename:
            raise ValueError(f"split must be one of {sorted(split_to_filename)}")

        dataset = RNADataset(
            csv_path=self.dataset_dir / split_to_filename[split],
            max_length=self.max_length,
            for_cnn=for_cnn,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )

    def get_class_names(self) -> List[str]:
        train_csv = self.dataset_dir / "train.csv"
        train_dataset = RNADataset(train_csv, max_length=self.max_length)
        return [
            train_dataset.idx_to_label[idx]
            for idx in sorted(train_dataset.idx_to_label)
        ]

    @property
    def input_dim(self) -> int:
        return len(RNADataset.channels) * self.max_length

    @property
    def num_classes(self) -> int:
        return len(self.get_class_names())
