"""
Pre-training dataset: loads unlabeled 5-channel .pkl files,
extracts ECG (ch0) and PPG (ch4), applies per-file Z-score normalization.
"""
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    Unlabeled multi-channel physiological signal dataset for JEPA pre-training.

    Each .pkl file contains:
        - data: (5, 3000) float32 array
        - timestamp: str

    We extract ECG (ch0) and PPG (ch4), normalize each independently per file.
    """

    def __init__(
        self,
        data_dir: str,
        channels: List[int] = None,
        normalize: str = "zscore",
        max_files: Optional[int] = None,
    ):
        """
        Args:
            data_dir: path to directory containing .pkl files
            channels: list of channel indices to keep (default: [0, 4] = ECG, PPG)
            normalize: normalization method ("zscore" or "minmax" or "none")
            max_files: limit number of files (for debugging)
        """
        if channels is None:
            channels = [0, 4]  # ECG, PPG

        self.data_dir = data_dir
        self.channels = channels
        self.normalize = normalize

        raw_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".pkl") and f.startswith("combined_processed_data")
        ])
        # Skip known corrupted file
        known_bad = {"combined_processed_data_2d_part10269_10.pkl"}
        self.files = [f for f in raw_files if f not in known_bad]

        if max_files is not None:
            self.files = self.files[:max_files]

        print(f"[PretrainDataset] Found {len(self.files)} files in {data_dir}")
        print(f"[PretrainDataset] Channels: {self.channels}, Normalize: {self.normalize}")

    def __len__(self) -> int:
        return len(self.files)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        """Per-channel Z-score normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (x - mean) / std

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        """Per-channel min-max normalization to [0, 1]."""
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)
        denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
        return (x - x_min) / denom

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ecg: (1, 3000) tensor — context signal
            ppg: (1, 3000) tensor — target signal
        """
        filepath = os.path.join(self.data_dir, self.files[idx])
        try:
            with open(filepath, "rb") as f:
                sample = pickle.load(f)

            data = sample["data"]  # (5, 3000)

            # Extract channels
            data = data[self.channels]  # (2, 3000): [ECG, PPG]

            # Per-file normalization
            if self.normalize == "zscore":
                data = self._zscore(data)
            elif self.normalize == "minmax":
                data = self._minmax(data)

            ecg = data[0:1]  # (1, 3000)
            ppg = data[1:2]  # (1, 3000)

            return torch.from_numpy(ecg.copy()).float(), torch.from_numpy(ppg.copy()).float()
        except Exception:
            # Fallback: return a random good sample
            fallback_idx = (idx + 1) % len(self.files)
            return self.__getitem__(fallback_idx)


class DownstreamDataset(Dataset):
    """
    Labeled single-channel dataset for downstream classification.

    Each .pkl file contains:
        - data: (1000,) float16
        - uid: str (patient id)
        - sampling_rate: int
        - label: [{"class": 0}] or [{"class": 1}]
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        split: str = "train",
        normalize: str = "zscore",
    ):
        """
        Args:
            data_dir: path to directory containing .pkl files
            split_file: path to train_test_split.json
            split: "train" or "test"
            normalize: normalization method
        """
        import json

        self.data_dir = data_dir
        self.normalize = normalize

        with open(split_file, "r") as f:
            split_data = json.load(f)

        self.files = split_data[split]
        print(f"[DownstreamDataset] {split}: {len(self.files)} files from {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        std = 1.0 if std == 0 else std
        return (x - mean) / std

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = os.path.join(self.data_dir, self.files[idx])
        with open(filepath, "rb") as f:
            sample = pickle.load(f)

        data = sample["data"].astype(np.float32)  # (1000,)

        # Normalize
        if self.normalize == "zscore":
            data = self._zscore(data)
        elif self.normalize == "minmax":
            d_min, d_max = data.min(), data.max()
            denom = 1.0 if d_max - d_min == 0 else d_max - d_min
            data = (data - d_min) / denom

        label = sample["label"][0]["class"]  # 0 or 1

        return torch.from_numpy(data).float().unsqueeze(0), label  # (1, 1000)
