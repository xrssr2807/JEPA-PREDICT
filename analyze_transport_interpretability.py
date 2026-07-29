"""
Validate and visualize the physiological causal-transport hypothesis.

The analysis is development-only by default. It uses the frozen validation
split and refuses to open the sealed test split unless explicitly overridden.
It produces segment- and patient-level metrics for:

1. physiological agreement with an ECG R-peak to PPG-foot PAT proxy;
2. token-dynamic transport versus segment-static and shuffled delay policies;
3. temporal-direction controls (zero delay, negative delay, reversed PPG);
4. pairing specificity (PPG shuffled across patients);
5. association between learned delay statistics and CHD labels.

This script validates causal consistency and physiological plausibility. It
does not claim causal discovery from observational waveforms.
"""

import argparse
import csv
import gc
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config
from dataset.data import MultiDiseaseDataset
from train_pretrain import (
    _checkpoint_uses_shared_private,
    _load_jepa_state_dict,
    build_model,
    seed_everything,
)


CHD_INDEX = 4
CONTROL_NAMES = (
    "dynamic_causal",
    "segment_static_delay",
    "token_shuffled_delay",
    "cross_patient_delay_policy",
    "fixed_prior",
    "zero_delay",
    "negative_delay",
    "reversed_ppg",
    "shuffled_pair",
)


class NamedMultiDiseaseDataset(Dataset):
    """Attach source filenames to the existing downstream dataset."""

    def __init__(self, dataset: MultiDiseaseDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        signals, labels, uid = self.dataset[index]
        return signals, labels, uid, self.dataset.files[index]


def _safe_float(value) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _mean_or_nan(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _std_or_nan(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if values.size > 1 else float("nan")


def _percentile(values: Sequence[float], q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rx = _average_ranks(x[valid])
    ry = _average_ranks(y[valid])
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    iterations: int,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_group_difference_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    rng: np.random.Generator,
    iterations: int,
) -> Tuple[float, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size < 2 or negative.size < 2:
        return float("nan"), float("nan")
    pos_idx = rng.integers(0, positive.size, size=(iterations, positive.size))
    neg_idx = rng.integers(0, negative.size, size=(iterations, negative.size))
    differences = positive[pos_idx].mean(axis=1) - negative[neg_idx].mean(axis=1)
    return (
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def _moving_average(signal: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return signal.astype(np.float64, copy=True)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(signal, kernel, mode="same")


def detect_r_peaks(ecg: np.ndarray, sample_rate: float) -> np.ndarray:
    """Polarity-aware QRS energy detector for quality-controlled PAT."""

    ecg = np.asarray(ecg, dtype=np.float64)
    if ecg.size < int(sample_rate * 2) or not np.isfinite(ecg).all():
        return np.empty(0, dtype=np.int64)
    centered = ecg - np.median(ecg)
    baseline = _moving_average(centered, max(3, round(sample_rate * 0.20)))
    qrs_signal = centered - baseline
    derivative = np.diff(qrs_signal, prepend=qrs_signal[0])
    energy = _moving_average(
        np.square(derivative), max(3, round(sample_rate * 0.12))
    )
    threshold = max(
        np.percentile(energy, 85.0),
        np.median(energy) + 2.5 * np.median(
            np.abs(energy - np.median(energy))
        ),
    )
    candidates = np.flatnonzero(
        (energy[1:-1] > energy[:-2])
        & (energy[1:-1] >= energy[2:])
        & (energy[1:-1] >= threshold)
    ) + 1
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)

    refractory = max(1, round(sample_rate * 0.30))
    selected = []
    for index in candidates[np.argsort(energy[candidates])[::-1]]:
        if all(abs(int(index) - previous) >= refractory for previous in selected):
            selected.append(int(index))

    refine_radius = max(1, round(sample_rate * 0.10))
    refined = []
    for index in selected:
        start = max(0, index - refine_radius)
        stop = min(qrs_signal.size, index + refine_radius + 1)
        local = np.abs(qrs_signal[start:stop])
        refined.append(start + int(np.argmax(local)))
    selected = np.asarray(sorted(set(refined)), dtype=np.int64)
    if selected.size < 2:
        return selected

    rr = np.diff(selected) / sample_rate
    valid_interval = (rr >= 0.30) & (rr <= 2.0)
    keep = np.ones(selected.size, dtype=bool)
    for index, valid in enumerate(valid_interval):
        if not valid:
            if index + 1 < selected.size - 1:
                keep[index + 1] = False
    return selected[keep]


def estimate_pat_proxy(
    ecg: np.ndarray,
    ppg: np.ndarray,
    sample_rate: float,
    min_delay_ms: float,
    max_delay_ms: float,
) -> Dict[str, object]:
    """
    Estimate ECG R-peak to PPG-foot pulse arrival time.

    The PPG foot is the local minimum immediately preceding the strongest
    positive upstroke within the physiological search window. This is a PAT
    proxy, not pure vascular PTT, because ECG-to-PPG timing includes PEP.
    """

    ecg = np.asarray(ecg, dtype=np.float64)
    ppg = np.asarray(ppg, dtype=np.float64)
    r_peaks = detect_r_peaks(ecg, sample_rate)
    ppg_smooth = _moving_average(ppg, max(3, round(sample_rate * 0.05)))
    derivative = np.gradient(ppg_smooth)
    min_samples = max(1, round(min_delay_ms * sample_rate / 1000.0))
    max_samples = max(min_samples + 1, round(max_delay_ms * sample_rate / 1000.0))
    backtrack = max(1, round(0.25 * sample_rate))

    foot_delays = []
    slope_delays = []
    foot_indices = []
    slope_indices = []
    derivative_scale = np.median(np.abs(derivative - np.median(derivative))) + 1e-8
    for peak_index, peak in enumerate(r_peaks):
        start = int(peak) + min_samples
        stop = min(ppg.size, int(peak) + max_samples + 1)
        if peak_index + 1 < r_peaks.size:
            stop = min(
                stop,
                int(r_peaks[peak_index + 1]) - max(1, round(0.08 * sample_rate)),
            )
        if stop - start < 3:
            continue
        slope = start + int(np.argmax(derivative[start:stop]))
        if derivative[slope] < derivative_scale:
            continue
        foot_start = max(start, slope - backtrack)
        foot_region = ppg_smooth[foot_start:slope + 1]
        minimum = float(foot_region.min())
        rise = max(float(ppg_smooth[slope]) - minimum, 1e-8)
        baseline = np.flatnonzero(foot_region <= minimum + 0.05 * rise)
        foot = (
            foot_start + int(baseline[-1])
            if baseline.size
            else foot_start + int(np.argmin(foot_region))
        )
        foot_delay = 1000.0 * (foot - int(peak)) / sample_rate
        slope_delay = 1000.0 * (slope - int(peak)) / sample_rate
        if min_delay_ms <= foot_delay <= max_delay_ms:
            foot_delays.append(foot_delay)
            slope_delays.append(slope_delay)
            foot_indices.append(foot)
            slope_indices.append(slope)

    result = {
        "r_peaks": r_peaks,
        "foot_indices": np.asarray(foot_indices, dtype=np.int64),
        "slope_indices": np.asarray(slope_indices, dtype=np.int64),
        "beat_count": len(foot_delays),
        "pat_foot_median_ms": float("nan"),
        "pat_foot_iqr_ms": float("nan"),
        "pat_slope_median_ms": float("nan"),
        "quality_pass": False,
    }
    if len(foot_delays) >= 3:
        foot_delays = np.asarray(foot_delays, dtype=np.float64)
        slope_delays = np.asarray(slope_delays, dtype=np.float64)
        foot_iqr = float(
            np.percentile(foot_delays, 75) - np.percentile(foot_delays, 25)
        )
        quality_pass = (
            foot_iqr <= 150.0
            and len(foot_delays) / max(1, len(r_peaks)) >= 0.5
        )
        result.update({
            "pat_foot_median_ms": (
                float(np.median(foot_delays))
                if quality_pass
                else float("nan")
            ),
            "pat_foot_iqr_ms": foot_iqr,
            "pat_slope_median_ms": (
                float(np.median(slope_delays))
                if quality_pass
                else float("nan")
            ),
            "quality_pass": quality_pass,
        })
    return result


def _cosine_distance_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    distance = 1.0 - (prediction * target).sum(dim=-1)
    weights = valid_rows.float()
    return (distance * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _shifted_targets(
    tokens: torch.Tensor,
    offsets: torch.Tensor,
    probabilities: torch.Tensor,
    direction: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mix tokens at source + direction * offset using per-token probabilities."""

    batch, length, dim = tokens.shape
    bins = offsets.numel()
    source = torch.arange(length, device=tokens.device).view(1, length, 1)
    target = source + int(direction) * offsets.view(1, 1, bins)
    valid = (target >= 0) & (target < length)
    weights = probabilities * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    indices = target.clamp(0, length - 1).expand(batch, -1, -1)
    expanded = tokens.unsqueeze(1).expand(-1, length, -1, -1)
    gathered = torch.gather(
        expanded,
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, dim),
    )
    mixed = (gathered * weights.unsqueeze(-1)).sum(dim=2)
    return mixed, valid.any(dim=-1).expand(batch, -1)


@torch.no_grad()
def transport_batch_diagnostics(
    model,
    ecg: torch.Tensor,
    ppg: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Extract model delay statistics and matched temporal controls."""

    ecg_input = model.context_encoder.tokenize(ecg)
    ppg_input = model.ppg_encoder.tokenize(ppg)
    length = min(ecg_input.size(1), ppg_input.size(1))
    ecg_input = ecg_input[:, :length]

    _, ecg_tokens = model.context_encoder.encode_tokens(
        ecg_input, return_all=True
    )
    _, ppg_teacher_tokens = model.target_encoder(ppg, return_all=True)
    ecg_online = model.ecg_token_proj(ecg_tokens)
    ppg_teacher = model.target_proj(ppg_teacher_tokens[:, :length])
    ecg_shared = ecg_online
    if model.phase2_shared_private_enabled:
        ecg_shared, _ = model.ecg_shared_private(ecg_online)
        ppg_teacher, _ = model.ppg_teacher_shared_private(ppg_teacher)

    prediction = model.ecg_to_ppg_predictor(ecg_shared)
    state = model._build_phase2_transport(ecg_shared)
    forward = state["forward_transport"]
    dynamic_target = torch.bmm(forward, ppg_teacher.float())

    offsets = model.phase2_delay_offsets
    conditional = state["conditional_delay_probabilities"]
    negative_target, negative_valid = _shifted_targets(
        ppg_teacher.float(), offsets, conditional, direction=-1
    )

    prior_offset = int(
        offsets[
            torch.argmin(
                (offsets.float() - float(model.phase2_delay_prior_tokens)).abs()
            )
        ].item()
    )
    fixed_probabilities = conditional.new_zeros(conditional.shape)
    prior_index = int((offsets == prior_offset).nonzero(as_tuple=False)[0].item())
    fixed_probabilities[..., prior_index] = 1.0
    fixed_target, fixed_valid = _shifted_targets(
        ppg_teacher.float(), offsets, fixed_probabilities, direction=1
    )

    segment_static_probabilities = conditional.mean(
        dim=1, keepdim=True
    ).expand_as(conditional)
    segment_static_target, _ = _shifted_targets(
        ppg_teacher.float(),
        offsets,
        segment_static_probabilities,
        direction=1,
    )
    token_shuffled_probabilities = torch.roll(
        conditional, shifts=max(1, length // 4), dims=1
    )
    token_shuffled_target, _ = _shifted_targets(
        ppg_teacher.float(),
        offsets,
        token_shuffled_probabilities,
        direction=1,
    )
    if ecg.size(0) > 1:
        cross_patient_probabilities = torch.roll(
            conditional, shifts=1, dims=0
        )
        cross_patient_delay_target, _ = _shifted_targets(
            ppg_teacher.float(),
            offsets,
            cross_patient_probabilities,
            direction=1,
        )
    else:
        cross_patient_delay_target = dynamic_target.new_full(
            dynamic_target.shape, float("nan")
        )

    max_offset = int(offsets.max().item())
    positions = torch.arange(length, device=ecg.device)
    interior = (
        (positions >= max_offset) & (positions < max(0, length - max_offset))
    ).view(1, -1).expand(ecg.size(0), -1)
    causal_valid = state["valid_rows"] & interior
    common_valid = causal_valid & negative_valid & fixed_valid
    if not common_valid.any():
        common_valid = state["valid_rows"]

    reversed_target = torch.bmm(
        forward, torch.flip(ppg_teacher.float(), dims=(1,))
    )
    if ecg.size(0) > 1:
        shuffled_teacher = torch.roll(ppg_teacher.float(), shifts=1, dims=0)
        shuffled_target = torch.bmm(forward, shuffled_teacher)
    else:
        shuffled_target = dynamic_target.new_full(dynamic_target.shape, float("nan"))

    scores = {
        "dynamic_causal": _cosine_distance_per_sample(
            prediction, dynamic_target, common_valid
        ),
        "segment_static_delay": _cosine_distance_per_sample(
            prediction, segment_static_target, common_valid
        ),
        "token_shuffled_delay": _cosine_distance_per_sample(
            prediction, token_shuffled_target, common_valid
        ),
        "cross_patient_delay_policy": _cosine_distance_per_sample(
            prediction, cross_patient_delay_target, common_valid
        ),
        "fixed_prior": _cosine_distance_per_sample(
            prediction, fixed_target, common_valid
        ),
        "zero_delay": _cosine_distance_per_sample(
            prediction, ppg_teacher.float(), common_valid
        ),
        "negative_delay": _cosine_distance_per_sample(
            prediction, negative_target, common_valid
        ),
        "reversed_ppg": _cosine_distance_per_sample(
            prediction, reversed_target, common_valid
        ),
        "shuffled_pair": _cosine_distance_per_sample(
            prediction, shuffled_target, common_valid
        ),
    }

    expected_ms = state["expected_delay"] * float(model.phase2_token_ms)
    valid = state["valid_rows"]
    valid_count = valid.sum(dim=1).clamp_min(1)
    delay_mean = (expected_ms * valid).sum(dim=1) / valid_count
    centered = (expected_ms - delay_mean.unsqueeze(1)).square() * valid
    delay_std = torch.sqrt(centered.sum(dim=1) / valid_count)

    positions_float = torch.arange(
        length, device=ecg.device, dtype=expected_ms.dtype
    ).view(1, -1)
    expected_target = positions_float + state["expected_delay"]
    pair_valid = valid[:, :-1] & valid[:, 1:]
    violations = (
        (expected_target[:, 1:] < expected_target[:, :-1]) & pair_valid
    ).sum(dim=1)
    violation_rate = violations.float() / pair_valid.sum(dim=1).clamp_min(1)
    smoothness_ms = (
        (expected_ms[:, 1:] - expected_ms[:, :-1]).abs() * pair_valid
    ).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1)
    entropy = -(
        conditional * conditional.clamp_min(1e-8).log()
    ).sum(dim=-1)
    entropy = (entropy * valid).sum(dim=1) / valid_count
    match_mass = (state["match_mass"] * valid).sum(dim=1) / valid_count

    return {
        **scores,
        "delay_mean_ms": delay_mean,
        "delay_std_ms": delay_std,
        "monotonic_violation_rate": violation_rate,
        "delay_smoothness_ms": smoothness_ms,
        "transport_entropy": entropy,
        "matched_mass": match_mass,
        "expected_delay_ms": expected_ms,
        "forward_transport": forward,
    }


def _load_model(checkpoint_path: str, device: torch.device):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    if int(checkpoint.get("pretrain_phase", 2)) != 2:
        raise ValueError("Transport interpretation requires a Phase 2 checkpoint")
    phase2_config = checkpoint.get("phase2_config", {})
    if phase2_config and not phase2_config.get("transport_enabled", True):
        raise ValueError("The supplied checkpoint has Transport disabled")

    config = Config()
    config.model.pretrain_phase = 2
    config.model.phase2_transport_enabled = True
    config.model.phase2_shared_private_enabled = _checkpoint_uses_shared_private(
        checkpoint
    )
    if phase2_config:
        config.model.phase2_sample_rate_hz = float(
            phase2_config.get("sample_rate_hz", config.model.phase2_sample_rate_hz)
        )
        config.model.phase2_min_delay_ms = float(
            phase2_config.get("min_delay_ms", config.model.phase2_min_delay_ms)
        )
        config.model.phase2_max_delay_ms = float(
            phase2_config.get("max_delay_ms", config.model.phase2_max_delay_ms)
        )
        config.model.phase2_delay_prior_ms = float(
            phase2_config.get("delay_prior_ms", config.model.phase2_delay_prior_ms)
        )
    model = build_model(config.model)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")
    _load_jepa_state_dict(model, checkpoint["model_state_dict"])
    checkpoint_metadata = {
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "pretrain_phase": checkpoint.get("pretrain_phase"),
        "phase2_config": dict(phase2_config),
    }
    del checkpoint
    gc.collect()
    model.to(device).eval()
    model._enforce_teacher_eval()
    return model, config, checkpoint_metadata


def _load_manifest(path: str, role: str, allow_test: bool) -> List[str]:
    if role == "test" and not allow_test:
        raise ValueError(
            "The test set is sealed. Use --role val, or pass --allow_test only "
            "after every model and baseline has been frozen."
        )
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if role not in manifest:
        raise KeyError(f"Split role '{role}' is absent from {path}")
    return list(manifest[role])


def _stratified_subset(
    dataset: MultiDiseaseDataset,
    max_segments: int,
    seed: int,
) -> List[str]:
    if max_segments <= 0 or max_segments >= len(dataset.files):
        return list(dataset.files)
    rng = random.Random(seed)
    files = list(dataset.files)
    rng.shuffle(files)
    return sorted(files[:max_segments])


def _uid_from_filename(filename: str) -> str:
    parts = filename.split("_")
    if parts[0] in {"train", "test", "val"} and len(parts) >= 3:
        return parts[1]
    return parts[0]


def _limit_segments_per_patient(
    files: Sequence[str],
    max_segments_per_patient: int,
    seed: int,
) -> List[str]:
    if max_segments_per_patient <= 0:
        return list(files)

    grouped = defaultdict(list)
    for filename in sorted(files):
        grouped[_uid_from_filename(filename)].append(filename)

    rng = random.Random(seed)
    selected = []
    for uid in sorted(grouped):
        patient_files = grouped[uid]
        if len(patient_files) > max_segments_per_patient:
            patient_files = rng.sample(
                patient_files, max_segments_per_patient
            )
        selected.extend(patient_files)
    return sorted(selected)


def _aggregate_patient_rows(segment_rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in segment_rows:
        grouped[row["uid"]].append(row)
    patient_rows = []
    numeric_fields = [
        "chd_label",
        "model_delay_mean_ms",
        "model_delay_std_ms",
        "pat_foot_median_ms",
        "pat_slope_median_ms",
        "pat_abs_error_ms",
        "monotonic_violation_rate",
        "delay_smoothness_ms",
        "transport_entropy",
        "matched_mass",
        *[f"{name}_loss" for name in CONTROL_NAMES],
    ]
    for uid, rows in sorted(grouped.items()):
        output = {"uid": uid, "segments": len(rows)}
        for field in numeric_fields:
            output[field] = _mean_or_nan(row.get(field, float("nan")) for row in rows)
        output["model_delay_between_segment_sd_ms"] = _std_or_nan(
            row["model_delay_mean_ms"] for row in rows
        )
        output["valid_pat_segments"] = sum(
            math.isfinite(float(row["pat_foot_median_ms"])) for row in rows
        )
        patient_rows.append(output)
    return patient_rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    segment_rows: List[dict],
    patient_rows: List[dict],
    model,
    config: Config,
    checkpoint: dict,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    valid_patients = [
        row for row in patient_rows
        if math.isfinite(float(row["pat_foot_median_ms"]))
    ]
    model_delay = [row["model_delay_mean_ms"] for row in valid_patients]
    pat_delay = [row["pat_foot_median_ms"] for row in valid_patients]
    differences = np.asarray(model_delay) - np.asarray(pat_delay)
    agreement = {
        "valid_patients": len(valid_patients),
        "model_delay_mean_ms": _mean_or_nan(model_delay),
        "model_delay_median_ms": _percentile(model_delay, 50),
        "pat_mean_ms": _mean_or_nan(pat_delay),
        "pat_median_ms": _percentile(pat_delay, 50),
        "spearman_model_vs_pat": spearman_correlation(model_delay, pat_delay),
        "mae_ms": _mean_or_nan(abs(value) for value in differences),
        "bias_model_minus_pat_ms": _mean_or_nan(differences),
        "bland_altman_lower_ms": (
            _mean_or_nan(differences) - 1.96 * _std_or_nan(differences)
        ),
        "bland_altman_upper_ms": (
            _mean_or_nan(differences) + 1.96 * _std_or_nan(differences)
        ),
    }

    controls = {}
    causal = np.asarray(
        [row["dynamic_causal_loss"] for row in patient_rows],
        dtype=np.float64,
    )
    for name in CONTROL_NAMES[1:]:
        control = np.asarray(
            [row[f"{name}_loss"] for row in patient_rows],
            dtype=np.float64,
        )
        gain = control - causal
        ci_low, ci_high = bootstrap_mean_ci(
            gain, rng, bootstrap_iterations
        )
        controls[name] = {
            "control_minus_dynamic_mean": _mean_or_nan(gain),
            "ci95": [ci_low, ci_high],
            "positive_supports_dynamic_causal": (
                math.isfinite(ci_low) and ci_low > 0.0
            ),
        }

    chd_positive = [
        row["model_delay_mean_ms"] for row in patient_rows
        if row["chd_label"] >= 0.5
    ]
    chd_negative = [
        row["model_delay_mean_ms"] for row in patient_rows
        if row["chd_label"] < 0.5
    ]
    group_ci = bootstrap_group_difference_ci(
        chd_positive, chd_negative, rng, bootstrap_iterations
    )
    pooled_sd = math.sqrt(
        0.5 * (
            np.nanvar(chd_positive, ddof=1)
            + np.nanvar(chd_negative, ddof=1)
        )
    ) if len(chd_positive) > 1 and len(chd_negative) > 1 else float("nan")
    chd_difference = _mean_or_nan(chd_positive) - _mean_or_nan(chd_negative)

    multi_segment = [
        row["model_delay_between_segment_sd_ms"]
        for row in patient_rows
        if row["segments"] >= 2
    ]
    effective_delays = (
        model.phase2_delay_offsets.detach().cpu().numpy()
        * float(model.phase2_token_ms)
    ).tolist()
    return {
        "protocol": {
            "analysis_unit": "patient",
            "test_set_sealed": True,
            "checkpoint_seed": checkpoint.get("seed"),
            "sample_rate_hz": config.model.phase2_sample_rate_hz,
            "requested_delay_range_ms": [
                config.model.phase2_min_delay_ms,
                config.model.phase2_max_delay_ms,
            ],
            "effective_token_delay_support_ms": effective_delays,
            "delay_prior_ms": config.model.phase2_delay_prior_ms,
            "pat_definition": "ECG R-peak to PPG foot; PAT proxy, not pure PTT",
            "causal_claim_scope": (
                "physiological temporal-direction consistency, not causal discovery"
            ),
        },
        "counts": {
            "segments": len(segment_rows),
            "patients": len(patient_rows),
            "quality_controlled_pat_segments": sum(
                int(row["pat_quality_pass"]) for row in segment_rows
            ),
            "chd_positive_patients": len(chd_positive),
            "chd_negative_patients": len(chd_negative),
            "segments_with_pat": sum(
                math.isfinite(float(row["pat_foot_median_ms"]))
                for row in segment_rows
            ),
        },
        "physiological_agreement": agreement,
        "dynamicity": {
            "patient_mean_delay_sd_ms": _std_or_nan(
                row["model_delay_mean_ms"] for row in patient_rows
            ),
            "median_within_patient_segment_sd_ms": _percentile(
                multi_segment, 50
            ),
            "median_within_segment_token_sd_ms": _percentile(
                [row["model_delay_std_ms"] for row in segment_rows], 50
            ),
            "median_monotonic_violation_rate": _percentile(
                [row["monotonic_violation_rate"] for row in segment_rows], 50
            ),
            "median_matched_mass": _percentile(
                [row["matched_mass"] for row in segment_rows], 50
            ),
        },
        "temporal_controls": controls,
        "chd_association": {
            "positive_mean_delay_ms": _mean_or_nan(chd_positive),
            "negative_mean_delay_ms": _mean_or_nan(chd_negative),
            "positive_minus_negative_ms": chd_difference,
            "ci95": list(group_ci),
            "standardized_mean_difference": (
                chd_difference / pooled_sd
                if math.isfinite(pooled_sd) and pooled_sd > 1e-8
                else float("nan")
            ),
            "interpretation_limit": (
                "association only; the downstream classifier does not consume "
                "the delay head directly"
            ),
        },
    }


def _format_number(value, digits: int = 4) -> str:
    value = float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def _write_report(path: Path, summary: dict) -> None:
    protocol = summary["protocol"]
    counts = summary["counts"]
    agreement = summary["physiological_agreement"]
    dynamicity = summary["dynamicity"]
    association = summary["chd_association"]
    lines = [
        "# Physiological Causal Transport Interpretability Report",
        "",
        "## Scope",
        "",
        "- Split: frozen development validation set; test set remains sealed.",
        f"- Segments/patients: {counts['segments']}/{counts['patients']}.",
        (
            "- PAT quality-controlled segments: "
            f"{counts['quality_controlled_pat_segments']}."
        ),
        (
            "- Effective token delay support: "
            + ", ".join(
                _format_number(value, 0)
                for value in protocol["effective_token_delay_support_ms"]
            )
            + " ms."
        ),
        (
            "- Physiological reference: ECG R-peak to PPG foot PAT proxy. "
            "This includes pre-ejection period and must not be called pure PTT."
        ),
        (
            "- Claim boundary: the controls test physiological temporal-direction "
            "consistency, not causal discovery from observational data."
        ),
        "",
        "## Physiological agreement",
        "",
        f"- Patients with valid PAT: {agreement['valid_patients']}.",
        (
            "- Model delay mean/median: "
            f"{_format_number(agreement['model_delay_mean_ms'], 1)} / "
            f"{_format_number(agreement['model_delay_median_ms'], 1)} ms."
        ),
        (
            "- PAT mean/median: "
            f"{_format_number(agreement['pat_mean_ms'], 1)} / "
            f"{_format_number(agreement['pat_median_ms'], 1)} ms."
        ),
        (
            "- Spearman(model delay, PAT): "
            + _format_number(agreement["spearman_model_vs_pat"])
        ),
        f"- MAE: {_format_number(agreement['mae_ms'], 1)} ms.",
        (
            "- Bias and Bland-Altman limits: "
            f"{_format_number(agreement['bias_model_minus_pat_ms'], 1)} ms "
            f"[{_format_number(agreement['bland_altman_lower_ms'], 1)}, "
            f"{_format_number(agreement['bland_altman_upper_ms'], 1)}]."
        ),
        "",
        "## Dynamic and monotonic behavior",
        "",
        (
            "- Between-patient SD of patient mean delay: "
            f"{_format_number(dynamicity['patient_mean_delay_sd_ms'], 1)} ms."
        ),
        (
            "- Median within-patient segment SD: "
            f"{_format_number(dynamicity['median_within_patient_segment_sd_ms'], 1)} ms."
        ),
        (
            "- Median within-segment token SD: "
            f"{_format_number(dynamicity['median_within_segment_token_sd_ms'], 1)} ms."
        ),
        (
            "- Median monotonic violation rate: "
            + _format_number(dynamicity["median_monotonic_violation_rate"])
        ),
        "",
        "## Temporal and pairing controls",
        "",
        "Positive values below mean that the control has higher cosine distance "
        "than learned dynamic causal transport.",
        "",
        "| Control | Control - dynamic | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for name, values in summary["temporal_controls"].items():
        lines.append(
            f"| {name} | {_format_number(values['control_minus_dynamic_mean'])} | "
            f"[{_format_number(values['ci95'][0])}, "
            f"{_format_number(values['ci95'][1])}] |"
        )
    lines.extend([
        "",
        "## CHD association",
        "",
        (
            "- CHD positive/negative mean delay: "
            f"{_format_number(association['positive_mean_delay_ms'], 1)} / "
            f"{_format_number(association['negative_mean_delay_ms'], 1)} ms."
        ),
        (
            "- Difference (positive - negative): "
            f"{_format_number(association['positive_minus_negative_ms'], 1)} ms "
            f"[{_format_number(association['ci95'][0], 1)}, "
            f"{_format_number(association['ci95'][1], 1)}]."
        ),
        (
            "- This is an association analysis. The downstream classifier uses "
            "the pretrained encoders, not the delay head itself."
        ),
        "",
        "## Paper decision rule",
        "",
        "The Transport claim is supported only if all of the following hold:",
        "",
        "1. learned delay has non-trivial agreement with waveform-derived PAT;",
        "2. dynamic transport beats fixed-delay and zero-delay controls;",
        "3. negative-delay, reversed-PPG and shuffled-pair controls are worse;",
        "4. monotonic violations remain near zero with non-degenerate matched mass;",
        "5. the downstream Transport-on advantage is reproduced across seeds.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_plots(
    output_dir: Path,
    segment_rows: List[dict],
    patient_rows: List[dict],
    examples: List[dict],
    summary: dict,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plots] matplotlib is unavailable; skipping figures")
        return

    valid = [
        row for row in patient_rows
        if math.isfinite(float(row["pat_foot_median_ms"]))
    ]
    if valid:
        x = np.asarray([row["pat_foot_median_ms"] for row in valid])
        y = np.asarray([row["model_delay_mean_ms"] for row in valid])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.scatter(x, y, s=16, alpha=0.45, color="#176B87", edgecolors="none")
        lower = min(x.min(), y.min())
        upper = max(x.max(), y.max())
        ax.plot([lower, upper], [lower, upper], "--", color="#B23A48", lw=1.4)
        ax.set_xlabel("Waveform PAT proxy (ms)")
        ax.set_ylabel("Model transport delay (ms)")
        ax.set_title("Physiological delay agreement")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "delay_agreement.png", dpi=220)
        plt.close(fig)

    controls = summary["temporal_controls"]
    names = list(controls)
    means = [controls[name]["control_minus_dynamic_mean"] for name in names]
    lows = [controls[name]["ci95"][0] for name in names]
    highs = [controls[name]["ci95"][1] for name in names]
    errors = np.asarray([
        [mean - low for mean, low in zip(means, lows)],
        [high - mean for mean, high in zip(means, highs)],
    ])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        np.arange(len(names)),
        means,
        color=["#2A9D8F", "#457B9D", "#E76F51", "#F4A261", "#6D597A"],
        yerr=errors,
        capsize=3,
    )
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(np.arange(len(names)), [name.replace("_", "\n") for name in names])
    ax.set_ylabel("Control loss - dynamic causal loss")
    ax.set_title("Temporal-direction and pairing controls")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "transport_controls.png", dpi=220)
    plt.close(fig)

    negative = [
        row["model_delay_mean_ms"] for row in patient_rows
        if row["chd_label"] < 0.5
    ]
    positive = [
        row["model_delay_mean_ms"] for row in patient_rows
        if row["chd_label"] >= 0.5
    ]
    if negative and positive:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        boxes = ax.boxplot(
            [negative, positive],
            tick_labels=["CHD negative", "CHD positive"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxes["boxes"], ["#A8DADC", "#E76F51"]):
            patch.set_facecolor(color)
        ax.set_ylabel("Patient mean transport delay (ms)")
        ax.set_title("Transport delay by CHD status")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "delay_by_chd.png", dpi=220)
        plt.close(fig)

    for index, example in enumerate(examples, start=1):
        ecg = example["ecg"]
        ppg = example["ppg"]
        pat = example["pat"]
        plan = example["plan"]
        delay = example["delay"]
        time = np.arange(ecg.size) / example["sample_rate"]
        fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0))
        axes[0].plot(time, ecg, color="#264653", lw=0.9, label="ECG")
        axes[0].scatter(
            pat["r_peaks"] / example["sample_rate"],
            ecg[pat["r_peaks"]],
            s=18,
            color="#E63946",
            label="R peak",
        )
        axes[0].legend(loc="upper right", frameon=False)
        axes[0].set_ylabel("ECG (z)")
        axes[1].plot(time, ppg, color="#2A9D8F", lw=0.9, label="PPG")
        feet = pat["foot_indices"]
        if feet.size:
            axes[1].scatter(
                feet / example["sample_rate"],
                ppg[feet],
                s=18,
                color="#F4A261",
                label="PPG foot",
            )
        axes[1].legend(loc="upper right", frameon=False)
        axes[1].set_ylabel("PPG (z)")
        image = axes[2].imshow(plan, origin="lower", aspect="auto", cmap="magma")
        axes[2].plot(
            np.arange(delay.size) + delay / example["token_ms"],
            np.arange(delay.size),
            color="#00E5FF",
            lw=1.2,
            label="expected path",
        )
        axes[2].set_xlabel("PPG target token")
        axes[2].set_ylabel("ECG source token")
        axes[2].legend(loc="upper left", frameon=False)
        fig.colorbar(image, ax=axes[2], label="transport mass")
        fig.suptitle(
            f"{example['uid']} | model delay={example['model_delay']:.0f} ms | "
            f"PAT={pat['pat_foot_median_ms']:.0f} ms"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"transport_example_{index}.png", dpi=220)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate physiological and causal Transport interpretation"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data_dir",
        default="/root/ppgchd/ppgchd/data_updated",
    )
    parser.add_argument(
        "--split",
        default="splits/multidisease_taskaware_downstream.json",
    )
    parser.add_argument("--role", choices=("train", "val", "test"), default="val")
    parser.add_argument("--allow_test", action="store_true")
    parser.add_argument("--output_dir", default="outputs_transport_interpretability")
    parser.add_argument("--max_segments", type=int, default=0)
    parser.add_argument(
        "--max_segments_per_patient",
        type=int,
        default=0,
        help=(
            "Deterministically cap segments per patient before any global "
            "max_segments subsampling; 0 keeps every segment"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--example_count", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap_iterations must be at least 100")
    seed_everything(args.seed, deterministic=True, enable_tf32=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Runtime] device={device} "
        f"torch_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()}"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _load_manifest(args.split, args.role, args.allow_test)
    files = _limit_segments_per_patient(
        files, args.max_segments_per_patient, args.seed
    )
    config = Config()
    base_dataset = MultiDiseaseDataset(
        data_dir=args.data_dir,
        split=args.role,
        disease_labels=config.data.multidisease_labels,
        normalize=config.data.normalize,
        normalize_clip=config.data.normalize_clip,
        channel="both",
        files=files,
    )
    subset_files = _stratified_subset(
        base_dataset, args.max_segments, args.seed
    )
    if len(subset_files) != len(base_dataset.files):
        base_dataset = MultiDiseaseDataset(
            data_dir=args.data_dir,
            split=args.role,
            disease_labels=config.data.multidisease_labels,
            normalize=config.data.normalize,
            normalize_clip=config.data.normalize_clip,
            channel="both",
            files=subset_files,
        )
    dataset = NamedMultiDiseaseDataset(base_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    model, model_config, checkpoint = _load_model(args.checkpoint, device)
    sample_rate = float(model_config.model.phase2_sample_rate_hz)
    min_delay = float(model_config.model.phase2_min_delay_ms)
    max_delay = float(model_config.model.phase2_max_delay_ms)
    segment_rows = []
    examples = []

    for batch_index, (signals, labels, uids, filenames) in enumerate(loader):
        if signals.size(1) < 2:
            raise ValueError("Dual-channel analysis requires PPG and ECG channels")
        ppg = signals[:, 0:1].to(device, non_blocking=True)
        ecg = signals[:, 1:2].to(device, non_blocking=True)
        diagnostics = transport_batch_diagnostics(model, ecg, ppg)
        cpu = {
            key: value.detach().cpu()
            for key, value in diagnostics.items()
        }
        ecg_numpy = ecg[:, 0].detach().cpu().numpy()
        ppg_numpy = ppg[:, 0].detach().cpu().numpy()

        for index, (uid, filename) in enumerate(zip(uids, filenames)):
            pat = estimate_pat_proxy(
                ecg_numpy[index],
                ppg_numpy[index],
                sample_rate,
                min_delay,
                max_delay,
            )
            model_delay = _safe_float(cpu["delay_mean_ms"][index])
            pat_delay = _safe_float(pat["pat_foot_median_ms"])
            row = {
                "uid": str(uid),
                "file": str(filename),
                "chd_label": _safe_float(labels[index, CHD_INDEX]),
                "model_delay_mean_ms": model_delay,
                "model_delay_std_ms": _safe_float(
                    cpu["delay_std_ms"][index]
                ),
                "pat_foot_median_ms": pat_delay,
                "pat_foot_iqr_ms": _safe_float(pat["pat_foot_iqr_ms"]),
                "pat_quality_pass": int(bool(pat["quality_pass"])),
                "r_peak_count": int(len(pat["r_peaks"])),
                "pat_slope_median_ms": _safe_float(
                    pat["pat_slope_median_ms"]
                ),
                "pat_beats": int(pat["beat_count"]),
                "pat_abs_error_ms": (
                    abs(model_delay - pat_delay)
                    if math.isfinite(pat_delay)
                    else float("nan")
                ),
                "monotonic_violation_rate": _safe_float(
                    cpu["monotonic_violation_rate"][index]
                ),
                "delay_smoothness_ms": _safe_float(
                    cpu["delay_smoothness_ms"][index]
                ),
                "transport_entropy": _safe_float(
                    cpu["transport_entropy"][index]
                ),
                "matched_mass": _safe_float(cpu["matched_mass"][index]),
            }
            for name in CONTROL_NAMES:
                row[f"{name}_loss"] = _safe_float(cpu[name][index])
            segment_rows.append(row)

            if (
                len(examples) < args.example_count
                and pat["beat_count"] >= 3
                and math.isfinite(pat_delay)
            ):
                examples.append({
                    "uid": str(uid),
                    "ecg": ecg_numpy[index].copy(),
                    "ppg": ppg_numpy[index].copy(),
                    "pat": pat,
                    "plan": cpu["forward_transport"][index].numpy(),
                    "delay": cpu["expected_delay_ms"][index].numpy(),
                    "model_delay": model_delay,
                    "sample_rate": sample_rate,
                    "token_ms": float(model.phase2_token_ms),
                })
        if (batch_index + 1) % 25 == 0:
            print(
                f"[TransportAnalysis] batches={batch_index + 1}/{len(loader)} "
                f"segments={len(segment_rows)}"
            )

    patient_rows = _aggregate_patient_rows(segment_rows)
    summary = _build_summary(
        segment_rows,
        patient_rows,
        model,
        model_config,
        checkpoint,
        args.bootstrap_iterations,
        args.seed,
    )
    summary["inputs"] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": os.path.abspath(args.split),
        "role": args.role,
        "data_dir": os.path.abspath(args.data_dir),
        "seed": args.seed,
    }

    _write_csv(output_dir / "segment_transport_metrics.csv", segment_rows)
    _write_csv(output_dir / "patient_transport_metrics.csv", patient_rows)
    with (output_dir / "transport_interpretability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    _write_report(
        output_dir / "transport_interpretability_report.md", summary
    )
    _make_plots(output_dir, segment_rows, patient_rows, examples, summary)
    print(
        "[Complete] Transport interpretation saved to "
        f"{output_dir.resolve()} | test_set_sealed={args.role != 'test'}"
    )


if __name__ == "__main__":
    main()
