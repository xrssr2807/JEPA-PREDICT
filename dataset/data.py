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


def compute_hrv_freq_features(signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    计算 HRV 频域特征 (无需 scipy, 用 numpy FFT)。

    频带定义:
      LF (低频): 0.04-0.15 Hz → 交感神经活动
      HF (高频): 0.15-0.40 Hz → 副交感神经活动
      LF/HF 比 → ⭐ CHD 强预警信号 (自主神经失衡)

    Args:
        signal: (L,) float32 信号
        fs: 采样率 (Hz)
    Returns:
        (5,) [LF_power, HF_power, LF_HF_ratio, total_power, peak_freq]
    """
    s = signal.astype(np.float64)
    n = len(s)

    # FFT 功率谱密度
    fft_vals = np.fft.rfft(s)
    power = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # 频带功率
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = np.sum(power[lf_mask]) if np.any(lf_mask) else 0.0
    hf_power = np.sum(power[hf_mask]) if np.any(hf_mask) else 0.0
    total_power = lf_power + hf_power
    lf_hf_ratio = lf_power / max(hf_power, 1e-12)

    # 峰值频率 (功率最大的频点)
    valid_range = (freqs >= 0.04) & (freqs <= 0.4)
    if np.any(valid_range):
        peak_idx = np.argmax(power[valid_range])
        peak_freq = freqs[valid_range][peak_idx]
    else:
        peak_freq = 0.0

    stats = np.array([lf_power, hf_power, lf_hf_ratio, total_power, peak_freq], dtype=np.float32)
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return stats


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


# ── 信号质量指数 (SQI) ─────────────────────────────────────
# 用于过滤低质量PPG，PhysioBridge + PPG BP综述均证实
# 信号质量是下游任务性能的关键前置环节

def compute_ppg_sqi(signal: np.ndarray, fs: int = 100) -> float:
    """
    计算PPG信号质量指数 (0~1, 1=最高质量)。

    基于三个维度：
    1. 幅值动态范围 — 信号有无足够的变化
    2. 峰值规律性 — 心跳是否规则 (HRV变异系数)
    3. 信噪比估计 — 信号相对噪声的强度

    Args:
        signal: (L,) float32 PPG信号
        fs: 采样率 (Hz)
    Returns:
        sqi: float in [0, 1]
    """
    s = signal.astype(np.float64)
    n = len(s)

    # 1. 幅值动态范围得分 (0~0.3)
    s_min, s_max = s.min(), s.max()
    dynamic_range = s_max - s_min
    range_score = min(dynamic_range / 0.5, 1.0) * 0.3  # 归一化到0.3

    # 2. 基于自相关的周期性得分 (0~0.4)
    # 用自相关检测PPG的周期性结构
    s_detrend = s - np.mean(s)
    if np.std(s_detrend) > 1e-8:
        autocorr = np.correlate(s_detrend, s_detrend, mode='same')
        mid = len(autocorr) // 2
        # 看自相关在合理心率范围内有无明显峰值 (50~120bpm → 0.5~2.0秒 → 50~200采样点@100Hz)
        search_range = autocorr[mid + fs // 4: mid + fs * 2]  # 0.25~2秒
        if len(search_range) > 0 and np.max(search_range) > 1e-8:
            # 归一化自相关峰值
            peak_val = np.max(search_range) / (np.std(s_detrend) ** 2 * n + 1e-10)
            periodicity_score = min(max(peak_val, 0), 1.0) * 0.4
        else:
            periodicity_score = 0.0
    else:
        periodicity_score = 0.0

    # 3. 基线稳定性得分 (0~0.3)
    # 检查信号的低频漂移程度
    smooth = np.convolve(s, np.ones(fs // 2) / (fs // 2), mode='same')
    baseline_var = np.std(smooth)
    baseline_score = max(0, 1.0 - min(baseline_var / 0.3, 1.0)) * 0.3

    sqi = range_score + periodicity_score + baseline_score
    return float(min(sqi, 1.0))


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
        binary_abnormal: bool = False,
        signal_quality_gate: float = 0.0,
        target_length: int = None,
        return_hrv: bool = False,  # ★ 返回 HRV 频域特征用于下游分类
    ):
        import json

        self.data_dir = data_dir
        self.normalize = normalize
        self.normalize_clip = normalize_clip
        self.binary_abnormal = binary_abnormal
        self.target_length = target_length
        self.signal_quality_gate = signal_quality_gate
        self.return_hrv = return_hrv

        with open(split_file, "r") as f:
            split_data = json.load(f)

        self.files = split_data[split]
        info = f"[DownstreamDataset] {split}: {len(self.files)} files from {data_dir}"
        info += f" (normalize={normalize})"
        if signal_quality_gate > 0:
            info += f" + SQI门控 ≥{signal_quality_gate:.2f}"
        print(info)
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

        # ★ 信号质量门控：在归一化前计算SQI
        if self.signal_quality_gate > 0:
            sqi = compute_ppg_sqi(data, fs=100)
            if sqi < self.signal_quality_gate:
                # 低于阈值 → 跳过此样本，返回下一个
                fallback_idx = (idx + 1) % len(self.files)
                return self.__getitem__(fallback_idx)

        # Normalize
        # ★ 信号对齐：先重采样，再归一化（保持预处理一致性）
        if self.target_length is not None and len(data) != self.target_length:
            from scipy.signal import resample
            data = resample(data, self.target_length).astype(np.float32)

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
        uid = str(sample.get("uid", f"unknown_{idx}"))  # 患者ID

        # ★ HRV 频域特征 (在归一化后的信号上计算)
        hrv_feats = None
        if self.return_hrv:
            hrv_feats = compute_hrv_freq_features(data, fs=100)
            hrv_feats = torch.from_numpy(hrv_feats).float()  # (5,)

        # Binary abnormal remapping: 0→0(normal), 1-5→1(abnormal)
        if self.binary_abnormal:
            label = 0 if label == 0 else 1

        if self.return_hrv:
            return torch.from_numpy(data).float().unsqueeze(0), label, uid, hrv_feats  # (1,1000), scalar, str, (5,)
        return torch.from_numpy(data).float().unsqueeze(0), label, uid  # (1, 1000), scalar, str


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


class DualDownstreamDataset(Dataset):
    """
    ★ 双通道下游数据集：加载配对的 ECG + PPG 用于 CHD 分类。

    假设 ECG 和 PPG 目录有相同文件名列表（配对数据）。
    复用两个 DownstreamDataset 实例，返回 (ecg, ppg, label, uid)。
    """

    def __init__(self, ppg_dataset: DownstreamDataset, ecg_dataset: DownstreamDataset):
        """
        Args:
            ppg_dataset: PPG DownstreamDataset (主数据集，提供 label + uid)
            ecg_dataset: ECG DownstreamDataset (提供 ECG 信号，与 PPG 配对)
        """
        assert len(ppg_dataset) == len(ecg_dataset), \
            f"PPG({len(ppg_dataset)}) 与 ECG({len(ecg_dataset)}) 样本数不匹配"
        self.ppg_dataset = ppg_dataset
        self.ecg_dataset = ecg_dataset

    def __len__(self) -> int:
        return len(self.ppg_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns:
            ecg: (1, 1000) tensor — ECG 信号
            ppg: (1, 1000) tensor — PPG 信号
            label: int — 分类标签
            uid: str — 患者ID
        """
        ppg_data, label, uid = self.ppg_dataset[idx]
        ecg_data, _, _ = self.ecg_dataset[idx]
        return ecg_data, ppg_data, label, uid
