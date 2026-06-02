"""
Pre-training dataset: loads unlabeled 5-channel .pkl files,
extracts ECG (ch0) and PPG (ch4), applies per-file normalization.

Supports: zscore, iqr (robust), minmax, none
"""
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_signal_stats(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute 16 physiological-statistics from a 1D signal (no scipy needed).

    Args:
        signal: (L,) float32 array (raw, before normalization)
        eps: small value to avoid division-by-zero
    Returns:
        (16,) float32 array of statistics
    """
    s = signal.astype(np.float64)
    n = len(s)

    # 1-4: Basic statistics
    mean = s.mean()
    std = s.std()
    s_min = s.min()
    s_max = s.max()

    # 5-6: Shape statistics
    z = (s - mean) / (std + eps)
    skewness = (z ** 3).mean()
    kurtosis = (z ** 4).mean() - 3.0  # excess kurtosis (normal=0)

    # 7: Signal energy
    energy = (s ** 2).sum() / n

    # 8: Zero-crossing rate (≈ dominant frequency indicator)
    zero_cross = np.sum(np.diff(s > 0)) / n

    # 9-13: Percentile values
    p5 = np.percentile(s, 5)
    p25 = np.percentile(s, 25)
    p50 = np.percentile(s, 50)
    p75 = np.percentile(s, 75)
    p95 = np.percentile(s, 95)

    # 14: Simple peak count (threshold = 0.5 * std above mean)
    threshold = 0.5 * std
    peaks = np.where(
        (s > mean + threshold) &
        (np.roll(s, 1) < s) &
        (np.roll(s, -1) < s)
    )[0]
    peak_count = len(peaks) / (n / 100.0)  # peaks per 100 samples (normalize by length)

    # 15: Mean peak amplitude (relative)
    if len(peaks) > 0:
        peak_amp_mean = (s[peaks] - mean).mean() / (std + eps)
    else:
        peak_amp_mean = 0.0

    # 16: Range
    value_range = s_max - s_min

    stats = np.array([
        mean, std, s_min, s_max,
        skewness, kurtosis, energy, zero_cross,
        p5, p25, p50, p75, p95,
        peak_count, peak_amp_mean, value_range,
    ], dtype=np.float32)

    # Replace inf/nan
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return stats


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
        normalize_clip: float = 10.0,
        max_files: Optional[int] = None,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        return_stats: bool = False,
    ):
        """
        Args:
            data_dir: path to directory containing .pkl files
            channels: list of channel indices to keep (default: [0, 4] = ECG, PPG)
            normalize: normalization method ("zscore", "iqr", "minmax", "none")
            normalize_clip: clip value after normalization (only for zscore/iqr)
            max_files: limit number of files (for debugging)
            augment: whether to apply PhysioAugment to ECG
            augment_config: dict of augmentation parameters
            return_stats: whether to return precomputed physiological statistics
        """
        if channels is None:
            channels = [0, 4]  # ECG, PPG

        self.data_dir = data_dir
        self.channels = channels
        self.normalize = normalize
        self.normalize_clip = normalize_clip
        self.augment = augment
        self.return_stats = return_stats

        if augment and augment_config:
            from .augment import PhysioAugment
            self.augmenter = PhysioAugment(**augment_config)
        else:
            self.augmenter = None

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
        print(f"[PretrainDataset] Channels: {self.channels}, Normalize: {self.normalize}"
              f"{' + clip=' + str(normalize_clip) if normalize in ('zscore','iqr') else ''}"
              f"{' + Augment' if augment else ''}")

    def __len__(self) -> int:
        return len(self.files)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        """Per-channel Z-score normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (x - mean) / std

    def _iqr(self, x: np.ndarray) -> np.ndarray:
        """Per-channel IQR robust normalization.

        Uses median and IQR instead of mean and std, making it robust to
        outliers (motion artifacts, electrode dropout) that plague Z-score.
        """
        median = np.median(x, axis=-1, keepdims=True)
        q25 = np.percentile(x, 25, axis=-1, keepdims=True)
        q75 = np.percentile(x, 75, axis=-1, keepdims=True)
        iqr_val = q75 - q25
        iqr_val = np.where(iqr_val < 1e-6, 1.0, iqr_val)
        return (x - median) / iqr_val

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        """Per-channel min-max normalization to [0, 1]."""
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)
        denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
        return (x - x_min) / denom

    def _normalize_signal(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization then clip."""
        if self.normalize == "zscore":
            x = self._zscore(x)
        elif self.normalize == "iqr":
            x = self._iqr(x)
        elif self.normalize == "minmax":
            x = self._minmax(x)
        # "none": no normalization

        if self.normalize in ("zscore", "iqr"):
            x = np.clip(x, -self.normalize_clip, self.normalize_clip)
        return x

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            ecg: (1, 3000) tensor
            ppg: (1, 3000) tensor
            ecg_stats: (16,) tensor (only if return_stats=True)
        """
        filepath = os.path.join(self.data_dir, self.files[idx])
        try:
            with open(filepath, "rb") as f:
                sample = pickle.load(f)

            data = sample["data"]  # (5, 3000)

            # Extract channels
            data = data[self.channels]  # (2, 3000): [ECG, PPG]

            # Compute stats on RAW signal before normalization
            ecg_stats = None
            if self.return_stats:
                ecg_stats = compute_signal_stats(data[0])
                ecg_stats = torch.from_numpy(ecg_stats)

            # Per-file normalization + clipping
            data = self._normalize_signal(data)

            ecg = data[0:1]  # (1, 3000)
            ppg = data[1:2]  # (1, 3000)

            # Apply PhysioAugment to ECG (context) only
            if self.augmenter is not None:
                ecg_aug = self.augmenter(ecg)
                ecg_tensor = torch.from_numpy(ecg_aug.copy()).float()
            else:
                ecg_tensor = torch.from_numpy(ecg.copy()).float()

            ppg_tensor = torch.from_numpy(ppg.copy()).float()

            if self.return_stats:
                return ecg_tensor, ppg_tensor, ecg_stats
            return ecg_tensor, ppg_tensor
        except Exception:
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
        normalize_clip: float = 10.0,
        binary_abnormal: bool = False,  # True: class 0→0, class 1-5→1
    ):
        """
        Args:
            data_dir: path to directory containing .pkl files
            split_file: path to train_test_split.json
            split: "train" or "test"
            normalize: normalization method ("zscore", "iqr", "minmax", "none")
            normalize_clip: clip value after zscore/iqr normalization
            binary_abnormal: if True, remap labels: 0→0(normal), 1-5→1(abnormal)
        """
        import json

        self.data_dir = data_dir
        self.normalize = normalize
        self.normalize_clip = normalize_clip
        self.binary_abnormal = binary_abnormal

        with open(split_file, "r") as f:
            split_data = json.load(f)

        self.files = split_data[split]
        print(f"[DownstreamDataset] {split}: {len(self.files)} files from {data_dir}"
              f" (normalize={normalize})")
        if binary_abnormal:
            print(f"[DownstreamDataset] Binary abnormal mode: class 0=正常, 1-5=异常")

    def __len__(self) -> int:
        return len(self.files)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        std = 1.0 if std == 0 else std
        return (x - mean) / std

    def _iqr(self, x: np.ndarray) -> np.ndarray:
        """IQR robust normalization (single-channel)."""
        median = np.median(x)
        q25 = np.percentile(x, 25)
        q75 = np.percentile(x, 75)
        iqr_val = q75 - q25
        iqr_val = 1.0 if iqr_val < 1e-6 else iqr_val
        return (x - median) / iqr_val

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        d_min, d_max = x.min(), x.max()
        denom = 1.0 if d_max - d_min == 0 else d_max - d_min
        return (x - d_min) / denom

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = os.path.join(self.data_dir, self.files[idx])
        with open(filepath, "rb") as f:
            sample = pickle.load(f)

        data = sample["data"].astype(np.float32)  # (1000,) or (1, 1000)

        # Handle both (1000,) and (1, 1000) shapes
        if data.ndim == 2:
            data = data.squeeze(0)  # (1, 1000) → (1000,)

        # Normalize
        if self.normalize == "zscore":
            data = self._zscore(data)
        elif self.normalize == "iqr":
            data = self._iqr(data)
        elif self.normalize == "minmax":
            data = self._minmax(data)
        # "none": no normalization

        if self.normalize in ("zscore", "iqr"):
            data = np.clip(data, -self.normalize_clip, self.normalize_clip)

        label = sample["label"][0]["class"]  # int

        # Binary abnormal remapping: 0→0(normal), 1-5→1(abnormal)
        if self.binary_abnormal:
            label = 0 if label == 0 else 1

        return torch.from_numpy(data).float().unsqueeze(0), label  # (1, 1000)


class PretrainDatasetPT(Dataset):
    """
    加载预处理好的 .pt 文件（已完成通道提取 + Z-score 归一化）。

    每个 .pt 文件包含:
        - ecg: (1, 3000) float32 tensor
        - ppg: (1, 3000) float32 tensor
    """

    def __init__(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
    ):
        """
        Args:
            data_dir: 预处理后 .pt 文件所在目录
            max_files: 限制文件数（调试用）
        """
        self.data_dir = data_dir

        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])

        if max_files is not None:
            self.files = self.files[:max_files]

        print(f"[PretrainDatasetPT] 找到 {len(self.files)} 个预处理文件于 {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ecg: (1, 3000) tensor — context signal
            ppg: (1, 3000) tensor — target signal
        """
        filepath = os.path.join(self.data_dir, self.files[idx])
        try:
            sample = torch.load(filepath, weights_only=True)
            return sample["ecg"], sample["ppg"]
        except Exception:
            fallback_idx = (idx + 1) % len(self.files)
            return self.__getitem__(fallback_idx)
