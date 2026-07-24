"""
Pre-training dataset: loads unlabeled 5-channel .pkl files,
extracts ECG (ch0) and PPG (ch4), applies per-file normalization.

Supports: zscore, iqr (robust), minmax, none
"""
import os
import pickle
import re
from typing import List, Tuple, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def infer_pretrain_uid(filename: str) -> str:
    """Infer a patient/recording UID from a pre-training filename.

    The current corpus uses names such as
    ``combined_processed_data_2d_part10269_10.pkl`` where ``part10269`` is
    the patient group and the final integer is a segment index.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(?:^|_)(part\d+)(?:_|$)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match = re.search(
        r"(?:^|_)((?:uid|patient|subject)[_-]?[a-z0-9]+)(?:_|$)",
        stem,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    # Best-effort fallback for ``recording_id_segmentIndex`` datasets.
    stripped = re.sub(r"_\d+$", "", stem)
    return stripped or stem


def split_pretrain_files(
    files: Sequence[str], val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Deterministically split files by UID so patients cannot leak."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    uid_to_files = {}
    for filename in sorted(files):
        uid_to_files.setdefault(infer_pretrain_uid(filename), []).append(filename)

    uids = sorted(uid_to_files)
    if len(uids) < 2:
        raise ValueError(
            "Patient-level pre-training validation needs at least two UIDs; "
            f"found {len(uids)} from {len(files)} files."
        )

    rng = np.random.default_rng(seed)
    shuffled_uids = list(uids)
    rng.shuffle(shuffled_uids)
    num_val = min(len(uids) - 1, max(1, round(len(uids) * val_ratio)))
    val_uids = set(shuffled_uids[:num_val])

    train_files = []
    val_files = []
    for uid in uids:
        target = val_files if uid in val_uids else train_files
        target.extend(uid_to_files[uid])

    return sorted(train_files), sorted(val_files)


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
        files: Optional[Sequence[str]] = None,
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
            files: optional explicit subset used by patient-level splitting
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

        raw_files = sorted(files) if files is not None else sorted([
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
        binary_abnormal: bool = False,  # True: class 0→0, class 1-5→1
        signal_quality_gate: float = 0.0,  # ★ SQI阈值：低于此值样本被跳过 (0=关闭)
        target_length: int = None,          # 信号对齐：插值到目标长度 (如1000→3000)
    ):
        """
        Args:
            ...
            signal_quality_gate: SQI阈值 (0~1), 低于此值样本被跳过 (0=关闭)
            target_length: 如果设置，线性插值到目标长度，匹配预训练输入尺度
        """
        import json

        self.data_dir = data_dir
        self.normalize = normalize
        self.normalize_clip = normalize_clip
        self.binary_abnormal = binary_abnormal
        self.target_length = target_length
        self.signal_quality_gate = signal_quality_gate

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

        # Binary abnormal remapping: 0→0(normal), 1-5→1(abnormal)
        if self.binary_abnormal:
            label = 0 if label == 0 else 1

        return torch.from_numpy(data).float().unsqueeze(0), label, uid  # (1, 1000), scalar, str


class PretrainDatasetPT(Dataset):
    """
    加载预处理好的 .pt 文件（已完成通道提取 + Z-score 归一化）。

    每个 .pt 文件包含:
        - ecg: (1, 3000) float32 tensor
        - ppg: (1, 3000) float32 tensor
        - ecg_stats: (16,) float32 tensor (Phase 0 preprocessing)
        - uid: patient/recording group identifier
    """

    def __init__(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
        return_stats: bool = False,
        files: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            data_dir: 预处理后 .pt 文件所在目录
            max_files: 限制文件数（调试用）
            return_stats: 是否要求并返回 ecg_stats
            files: 可选的显式文件子集
        """
        self.data_dir = data_dir
        self.return_stats = return_stats

        self.files = sorted(files) if files is not None else sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])

        if max_files is not None:
            self.files = self.files[:max_files]

        print(f"[PretrainDatasetPT] 找到 {len(self.files)} 个预处理文件于 {data_dir}")
        if self.return_stats and self.files:
            first_path = os.path.join(self.data_dir, self.files[0])
            first_sample = torch.load(first_path, map_location="cpu", weights_only=True)
            if "ecg_stats" not in first_sample:
                raise RuntimeError(
                    "Stats loss is enabled, but preprocessed files do not contain "
                    f"'ecg_stats' (checked {first_path}). Re-run: "
                    "python preprocess.py --overwrite"
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            ecg: (1, 3000) tensor — context signal
            ppg: (1, 3000) tensor — target signal
            ecg_stats: (16,) tensor when return_stats=True
        """
        filepath = os.path.join(self.data_dir, self.files[idx])
        try:
            sample = torch.load(filepath, weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load pre-training file: {filepath}") from exc

        for key in ("ecg", "ppg"):
            if key not in sample:
                raise RuntimeError(f"Missing '{key}' in {filepath}")
            if not torch.isfinite(sample[key]).all():
                raise RuntimeError(
                    f"Non-finite values found in '{key}' at {filepath}; "
                    "re-run preprocess.py --overwrite with the Phase 0 code"
                )

        if self.return_stats:
            if "ecg_stats" not in sample:
                raise RuntimeError(
                    f"Missing ecg_stats in {filepath}; re-run preprocess.py --overwrite"
                )
            if not torch.isfinite(sample["ecg_stats"]).all():
                raise RuntimeError(
                    f"Non-finite values found in 'ecg_stats' at {filepath}; "
                    "re-run preprocess.py --overwrite with the Phase 0 code"
                )
            return sample["ecg"], sample["ppg"], sample["ecg_stats"]
        return sample["ecg"], sample["ppg"]


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


MULTIDISEASE_LABEL_SOURCES = {
    "其他疾病": ("下肢动脉硬化闭塞症", "脑卒中（中风）"),
}


def multidisease_label_value(label_dict: dict, label_name: str) -> float:
    """Resolve one downstream target, including explicitly merged labels."""
    source_names = MULTIDISEASE_LABEL_SOURCES.get(label_name, (label_name,))
    return float(any(bool(label_dict.get(name, 0)) for name in source_names))


class MultiDiseaseDataset(Dataset):
    """
    Multi-label downstream dataset for the configured disease labels.

    Files are split by filename prefix: train_<uid>_<segment>.pkl / test_<uid>_<segment>.pkl.
    Each sample contains:
        - data: (2, 1000) or (1000,) float signal
        - uid: patient id
        - label: dict disease_name -> 0/1

    By default this returns one channel so it can reuse the existing 1-channel
    pretrained encoder. Set channel=None only for models that explicitly accept
    multi-channel input.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        disease_labels: Optional[List[str]] = None,
        normalize: str = "zscore",
        normalize_clip: float = 10.0,
        channel: Optional[int] = 0,
        target_length: int = None,
        files: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.normalize_clip = normalize_clip
        if isinstance(channel, str):
            self.channel = None if channel == "both" else int(channel)
        else:
            self.channel = channel
        self.target_length = target_length
        self.disease_labels = disease_labels or [
            "高血压", "高血糖", "高血脂", "其他疾病", "冠心病",
            "心律失常（房颤、频发早搏等）", "糖尿病", "颈动脉斑块",
        ]

        prefix = split + "_"
        if files is None:
            self.files = sorted([
                f for f in os.listdir(data_dir)
                if f.endswith(".pkl") and f.startswith(prefix)
            ])
        else:
            self.files = sorted(files)
        print(f"[MultiDiseaseDataset] {split}: {len(self.files)} files from {data_dir}")
        print(f"[MultiDiseaseDataset] channel={channel} labels={self.disease_labels}")

    def __len__(self) -> int:
        return len(self.files)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
        return (x - mean) / std

    def _iqr(self, x: np.ndarray) -> np.ndarray:
        median = np.median(x, axis=-1, keepdims=True)
        q25 = np.percentile(x, 25, axis=-1, keepdims=True)
        q75 = np.percentile(x, 75, axis=-1, keepdims=True)
        iqr_val = q75 - q25
        iqr_val = np.where(np.isfinite(iqr_val) & (iqr_val >= 1e-6), iqr_val, 1.0)
        return (x - median) / iqr_val

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)
        denom = x_max - x_min
        denom = np.where(np.isfinite(denom) & (denom >= 1e-6), denom, 1.0)
        return (x - x_min) / denom

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filepath = os.path.join(self.data_dir, self.files[idx])
        with open(filepath, "rb") as f:
            sample = pickle.load(f)

        data = sample["data"].astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.ndim == 1:
            data = data[None, :]
        elif self.channel is not None:
            data = data[self.channel:self.channel + 1]

        if self.target_length is not None and data.shape[-1] != self.target_length:
            from scipy.signal import resample
            data = resample(data, self.target_length, axis=-1).astype(np.float32)

        if self.normalize == "zscore":
            data = self._zscore(data)
        elif self.normalize == "iqr":
            data = self._iqr(data)
        elif self.normalize == "minmax":
            data = self._minmax(data)

        if self.normalize in ("zscore", "iqr"):
            data = np.clip(data, -self.normalize_clip, self.normalize_clip)
        data = np.nan_to_num(data, nan=0.0, posinf=self.normalize_clip, neginf=-self.normalize_clip)

        label_dict = sample["label"]
        labels = np.array(
            [
                multidisease_label_value(label_dict, name)
                for name in self.disease_labels
            ],
            dtype=np.float32,
        )
        uid = str(sample.get("uid", self.files[idx].split("_")[1]))

        return torch.from_numpy(data.copy()).float(), torch.from_numpy(labels), uid


class MultiDiseasePatientMILDataset(Dataset):
    """
    Patient-level multi-instance dataset.

    Each item is a fixed-size bag of segments from one UID:
        segments: (S, C, L)
        labels:   (num_labels,)
        uid:      str
        mask:     (S,), True for real segments and False for padding
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        disease_labels: Optional[List[str]] = None,
        normalize: str = "zscore",
        normalize_clip: float = 10.0,
        channel: Optional[int] = 0,
        target_length: int = None,
        max_segments: int = 8,
        files: Optional[List[str]] = None,
        train: bool = True,
    ):
        self.segment_dataset = MultiDiseaseDataset(
            data_dir=data_dir,
            split=split,
            disease_labels=disease_labels,
            normalize=normalize,
            normalize_clip=normalize_clip,
            channel=channel,
            target_length=target_length,
            files=files,
        )
        self.max_segments = max_segments
        self.train = train
        self.uid_to_indices = {}
        for idx, fname in enumerate(self.segment_dataset.files):
            uid = self._uid_from_file(fname)
            self.uid_to_indices.setdefault(uid, []).append(idx)
        self.uids = sorted(self.uid_to_indices)
        self.files = [
            self.segment_dataset.files[self.uid_to_indices[uid][0]]
            for uid in self.uids
        ]
        self.data_dir = data_dir
        self.disease_labels = self.segment_dataset.disease_labels
        print(
            f"[MultiDiseasePatientMILDataset] {split}: {len(self.uids)} patients, "
            f"max_segments={max_segments}"
        )

    @staticmethod
    def _uid_from_file(fname: str) -> str:
        parts = fname.split("_")
        if parts[0] in {"train", "test", "val"} and len(parts) >= 3:
            return parts[1]
        return parts[0]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        uid = self.uids[idx]
        indices = self.uid_to_indices[uid]
        if len(indices) >= self.max_segments:
            if self.train:
                chosen = np.random.choice(indices, self.max_segments, replace=False)
            else:
                chosen = np.linspace(0, len(indices) - 1, self.max_segments).round().astype(int)
                chosen = [indices[i] for i in chosen]
            valid_count = self.max_segments
        else:
            # Repeating a short patient's segments makes MIL attention treat
            # duplicated evidence as independent observations. Keep each real
            # segment once and pad the rest; the returned mask excludes padding.
            chosen = list(indices)
            if self.train:
                chosen = list(np.random.permutation(chosen))
            valid_count = len(chosen)

        segments = []
        label = None
        for segment_idx in chosen:
            x, y, _ = self.segment_dataset[int(segment_idx)]
            segments.append(x)
            if label is None:
                label = y

        if valid_count < self.max_segments:
            padding = [torch.zeros_like(segments[0]) for _ in range(self.max_segments - valid_count)]
            segments.extend(padding)

        segment_mask = torch.zeros(self.max_segments, dtype=torch.bool)
        segment_mask[:valid_count] = True
        return torch.stack(segments, dim=0), label, uid, segment_mask
