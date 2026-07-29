#!/usr/bin/env python3
"""Controlled time-shift intervention for Phase 2 ECG-to-PPG Transport.

The Phase 2 delay policy is conditioned on ECG tokens only. Consequently, a
PPG intervention cannot change the delay-head output without changing the
architecture. This analysis instead tests the claim the current model can
actually support:

1. shifting PPG away from its observed timing should worsen alignment under
   the original ECG-conditioned Transport plan;
2. shifting the Transport target columns by the known intervention should
   recover alignment quality; and
3. a loss-profile search over candidate compensations should recover the
   direction and magnitude of the injected temporal shift.

The default protocol is validation-only and keeps the test set sealed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analyze_transport_interpretability import (
    NamedMultiDiseaseDataset,
    _limit_segments_per_patient,
    _load_manifest,
    _load_model,
    _safe_float,
    _stratified_subset,
    bootstrap_mean_ci,
    spearman_correlation,
)
from config import Config
from dataset.data import MultiDiseaseDataset
from train_downstream import seed_everything


def _mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _std_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size > 1 else float("nan")


def _format_number(value: float, digits: int = 4) -> str:
    value = float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shift_label(value: float) -> str:
    rounded = round(float(value), 6)
    text = f"{abs(rounded):g}".replace(".", "p")
    return ("m" if rounded < 0 else "p") + text


def shift_waveform_non_circular(
    waveform: torch.Tensor,
    shift_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift ``[B, C, T]`` without wraparound; positive shifts delay/right."""

    if waveform.ndim != 3:
        raise ValueError(f"Expected [B,C,T], got {tuple(waveform.shape)}")
    length = waveform.size(-1)
    shift = int(shift_samples)
    output = torch.zeros_like(waveform)
    valid = torch.zeros(
        (waveform.size(0), length),
        dtype=torch.bool,
        device=waveform.device,
    )
    if abs(shift) >= length:
        return output, valid
    if shift == 0:
        output.copy_(waveform)
        valid.fill_(True)
    elif shift > 0:
        output[..., shift:] = waveform[..., : length - shift]
        valid[:, shift:] = True
    else:
        amount = -shift
        output[..., : length - amount] = waveform[..., amount:]
        valid[:, : length - amount] = True
    return output, valid


def shift_sequence_non_circular(
    sequence: torch.Tensor,
    shift_tokens: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linearly shift ``[B, L, D]`` without wraparound.

    A positive value delays the sequence to the right. Invalid boundary
    positions are zero and marked False.
    """

    if sequence.ndim != 3:
        raise ValueError(f"Expected [B,L,D], got {tuple(sequence.shape)}")
    batch, length, dim = sequence.shape
    shift = float(shift_tokens)
    output_positions = torch.arange(
        length, device=sequence.device, dtype=torch.float32
    )
    source_positions = output_positions - shift
    lower = torch.floor(source_positions).long()
    upper = lower + 1
    fraction = (source_positions - lower.float()).to(sequence.dtype)
    valid = (source_positions >= 0.0) & (source_positions <= length - 1)

    lower_indices = lower.clamp(0, length - 1)
    upper_indices = upper.clamp(0, length - 1)
    lower_values = sequence[:, lower_indices, :]
    upper_values = sequence[:, upper_indices, :]
    shifted = (
        lower_values * (1.0 - fraction).view(1, length, 1)
        + upper_values * fraction.view(1, length, 1)
    )
    shifted = shifted * valid.to(sequence.dtype).view(1, length, 1)
    return shifted, valid.view(1, length).expand(batch, -1)


def shift_transport_columns(
    plan: torch.Tensor,
    shift_tokens: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move Transport mass along target columns without circular wrapping."""

    if plan.ndim != 3 or plan.size(1) != plan.size(2):
        raise ValueError(f"Expected square [B,L,L] plan, got {tuple(plan.shape)}")
    batch, rows, columns = plan.shape
    flattened = plan.reshape(batch * rows, columns, 1)
    shifted, _ = shift_sequence_non_circular(flattened, shift_tokens)
    shifted = shifted.reshape(batch, rows, columns)
    row_mass = shifted.sum(dim=-1)
    normalized = shifted / row_mass.unsqueeze(-1).clamp_min(1e-8)
    normalized = torch.where(
        (row_mass > 0).unsqueeze(-1), normalized, torch.zeros_like(normalized)
    )
    return normalized, row_mass


def waveform_mask_to_token_mask(
    valid_waveform: torch.Tensor,
    token_length: int,
    boundary_margin_tokens: int,
) -> torch.Tensor:
    """Conservatively project a waveform-valid mask onto encoder tokens."""

    if valid_waveform.ndim != 2:
        raise ValueError(
            f"Expected [B,T] waveform mask, got {tuple(valid_waveform.shape)}"
        )
    coverage = F.interpolate(
        valid_waveform.float().unsqueeze(1),
        size=int(token_length),
        mode="area",
    ).squeeze(1)
    valid = coverage >= 1.0 - 1e-6
    margin = max(0, int(boundary_margin_tokens))
    if margin > 0:
        invalid = (~valid).float().unsqueeze(1)
        invalid = F.max_pool1d(
            invalid,
            kernel_size=2 * margin + 1,
            stride=1,
            padding=margin,
        )
        valid = invalid.squeeze(1) < 0.5
        valid[:, :margin] = False
        valid[:, -margin:] = False
    return valid


def _masked_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    distance = 1.0 - (prediction * target).sum(dim=-1)
    weights = valid_rows.float()
    counts = weights.sum(dim=1)
    score = (distance * weights).sum(dim=1) / counts.clamp_min(1.0)
    return torch.where(
        counts > 0,
        score,
        torch.full_like(score, float("nan")),
    )


@torch.no_grad()
def evaluate_compensation_profile(
    prediction: torch.Tensor,
    shifted_teacher: torch.Tensor,
    forward_plan: torch.Tensor,
    source_valid: torch.Tensor,
    teacher_valid: torch.Tensor,
    candidate_shifts_tokens: Sequence[float],
    mass_tolerance: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Evaluate per-sample loss for candidate target-column compensations."""

    if prediction.shape != shifted_teacher.shape:
        raise ValueError(
            f"Prediction/teacher mismatch: {prediction.shape} vs "
            f"{shifted_teacher.shape}"
        )
    losses = []
    valid_fractions = []
    overlap_masses = []
    for candidate in candidate_shifts_tokens:
        candidate_plan, retained_mass = shift_transport_columns(
            forward_plan, float(candidate)
        )
        teacher_overlap = torch.bmm(
            candidate_plan,
            teacher_valid.float().unsqueeze(-1),
        ).squeeze(-1)
        valid_rows = (
            source_valid
            & (retained_mass >= 1.0 - float(mass_tolerance))
            & (teacher_overlap >= 1.0 - float(mass_tolerance))
        )
        target = torch.bmm(candidate_plan, shifted_teacher.float())
        losses.append(_masked_cosine_distance(prediction, target, valid_rows))
        valid_fractions.append(valid_rows.float().mean(dim=1))
        overlap_masses.append(
            (teacher_overlap * source_valid.float()).sum(dim=1)
            / source_valid.float().sum(dim=1).clamp_min(1.0)
        )
    return {
        "candidate_shifts_tokens": [float(value) for value in candidate_shifts_tokens],
        "losses": torch.stack(losses, dim=1),
        "valid_fractions": torch.stack(valid_fractions, dim=1),
        "overlap_masses": torch.stack(overlap_masses, dim=1),
    }


@torch.no_grad()
def _encode_ecg_transport(model, ecg: torch.Tensor):
    ecg_input = model.context_encoder.tokenize(ecg)
    _, ecg_tokens = model.context_encoder.encode_tokens(
        ecg_input, return_all=True
    )
    ecg_online = model.ecg_token_proj(ecg_tokens)
    ecg_shared = ecg_online
    if model.phase2_shared_private_enabled:
        ecg_shared, _ = model.ecg_shared_private(ecg_online)
    prediction = model.ecg_to_ppg_predictor(ecg_shared)
    state = model._build_phase2_transport(ecg_shared)
    return prediction.float(), state


@torch.no_grad()
def _encode_ppg_teacher(model, ppg: torch.Tensor, length: int) -> torch.Tensor:
    _, tokens = model.target_encoder(ppg, return_all=True)
    teacher = model.target_proj(tokens[:, :length])
    if model.phase2_shared_private_enabled:
        teacher, _ = model.ppg_teacher_shared_private(teacher)
    return teacher.float()


def _best_candidate(
    losses: torch.Tensor,
    candidates_ms: Sequence[float],
) -> Tuple[List[float], List[float]]:
    recovered = []
    minimum = []
    for row in losses.detach().cpu():
        finite = torch.isfinite(row)
        if not bool(finite.any()):
            recovered.append(float("nan"))
            minimum.append(float("nan"))
            continue
        safe = torch.where(finite, row, torch.full_like(row, float("inf")))
        index = int(torch.argmin(safe).item())
        recovered.append(float(candidates_ms[index]))
        minimum.append(float(row[index]))
    return recovered, minimum


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _attach_zero_shift_deltas(rows: List[dict]) -> None:
    baseline = {}
    for row in rows:
        if abs(float(row["injected_shift_ms"])) < 1e-8:
            baseline[row["file"]] = float(row["uncompensated_loss"])
    for row in rows:
        zero = baseline.get(row["file"], float("nan"))
        row["zero_shift_loss"] = zero
        row["loss_delta_vs_zero"] = (
            float(row["uncompensated_loss"]) - zero
            if math.isfinite(zero)
            and math.isfinite(float(row["uncompensated_loss"]))
            else float("nan")
        )


def _aggregate_patient_shift_rows(segment_rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in segment_rows:
        grouped[(row["uid"], float(row["injected_shift_ms"]))].append(row)
    fields = [
        "uncompensated_loss",
        "oracle_compensated_loss",
        "oracle_compensation_benefit",
        "minimum_profile_loss",
        "recovered_shift_ms",
        "recovery_abs_error_ms",
        "loss_delta_vs_zero",
        "uncompensated_valid_fraction",
        "oracle_valid_fraction",
        "uncompensated_overlap_mass",
        "oracle_overlap_mass",
    ]
    outputs = []
    for (uid, shift), rows in sorted(grouped.items()):
        output = {
            "uid": uid,
            "injected_shift_ms": shift,
            "segments": len(rows),
        }
        for field in fields:
            output[field] = _mean_or_nan(row[field] for row in rows)
        outputs.append(output)
    return outputs


def _patient_response_statistics(
    patient_rows: List[dict],
    shifts_ms: Sequence[float],
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    by_uid = defaultdict(dict)
    for row in patient_rows:
        by_uid[row["uid"]][float(row["injected_shift_ms"])] = row
    required = {float(value) for value in shifts_ms}
    complete = {
        uid: values
        for uid, values in by_uid.items()
        if required.issubset(values)
    }

    shift_summary = {}
    for shift in sorted(required):
        rows = [values[shift] for values in complete.values()]
        delta = np.asarray(
            [row["loss_delta_vs_zero"] for row in rows], dtype=np.float64
        )
        benefit = np.asarray(
            [row["oracle_compensation_benefit"] for row in rows],
            dtype=np.float64,
        )
        recovered = np.asarray(
            [row["recovered_shift_ms"] for row in rows], dtype=np.float64
        )
        delta_ci = bootstrap_mean_ci(delta, rng, iterations)
        benefit_ci = bootstrap_mean_ci(benefit, rng, iterations)
        shift_summary[str(shift)] = {
            "patients": len(rows),
            "uncompensated_loss_mean": _mean_or_nan(
                row["uncompensated_loss"] for row in rows
            ),
            "loss_delta_vs_zero_mean": _mean_or_nan(delta),
            "loss_delta_vs_zero_ci95": list(delta_ci),
            "oracle_compensation_benefit_mean": _mean_or_nan(benefit),
            "oracle_compensation_benefit_ci95": list(benefit_ci),
            "recovered_shift_mean_ms": _mean_or_nan(recovered),
            "recovery_mae_ms": _mean_or_nan(
                abs(value - shift) for value in recovered
            ),
        }

    patient_slopes = []
    patient_correlations = []
    x = np.asarray(sorted(required), dtype=np.float64)
    for values in complete.values():
        y = np.asarray(
            [values[shift]["recovered_shift_ms"] for shift in x],
            dtype=np.float64,
        )
        if np.isfinite(y).all() and np.unique(x).size >= 2:
            patient_slopes.append(float(np.polyfit(x, y, 1)[0]))
            patient_correlations.append(spearman_correlation(x, y))
    slope_ci = bootstrap_mean_ci(patient_slopes, rng, iterations)

    zero_minimum = []
    for values in complete.values():
        losses = {
            shift: float(values[shift]["uncompensated_loss"])
            for shift in x
        }
        finite = {key: value for key, value in losses.items() if math.isfinite(value)}
        if finite:
            minimum_shift = min(finite, key=finite.get)
            zero_minimum.append(float(abs(minimum_shift) < 1e-8))

    return {
        "complete_patients": len(complete),
        "by_injected_shift": shift_summary,
        "recovery": {
            "patient_slope_mean": _mean_or_nan(patient_slopes),
            "patient_slope_ci95": list(slope_ci),
            "patient_spearman_mean": _mean_or_nan(patient_correlations),
            "patient_recovery_mae_ms": _mean_or_nan(
                row["recovery_abs_error_ms"] for row in patient_rows
            ),
            "zero_shift_is_minimum_rate": _mean_or_nan(zero_minimum),
        },
    }


def _write_report(path: Path, summary: dict) -> None:
    protocol = summary["protocol"]
    response = summary["response"]
    lines = [
        "# Transport time-shift intervention",
        "",
        f"- Domain: `{protocol['shift_domain']}`.",
        f"- Role: `{protocol['role']}`; test sealed: "
        f"`{str(protocol['test_set_sealed']).lower()}`.",
        f"- Segments/patients: {protocol['segments']}/"
        f"{response['complete_patients']}.",
        f"- Token duration: {_format_number(protocol['token_ms'], 1)} ms.",
        "- Delay-head scope: ECG-conditioned only. Recovered shift is the "
        "argmin of a pairwise compensation-loss profile, not delay-head output.",
        "",
        "## Shift sensitivity and compensation",
        "",
        "| Injected shift | Loss delta vs zero | 95% CI | "
        "Oracle compensation benefit | 95% CI | Recovered shift | Recovery MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(
        response["by_injected_shift"].items(), key=lambda pair: float(pair[0])
    ):
        lines.append(
            f"| {float(key):.0f} ms | "
            f"{_format_number(item['loss_delta_vs_zero_mean'])} | "
            f"[{_format_number(item['loss_delta_vs_zero_ci95'][0])}, "
            f"{_format_number(item['loss_delta_vs_zero_ci95'][1])}] | "
            f"{_format_number(item['oracle_compensation_benefit_mean'])} | "
            f"[{_format_number(item['oracle_compensation_benefit_ci95'][0])}, "
            f"{_format_number(item['oracle_compensation_benefit_ci95'][1])}] | "
            f"{_format_number(item['recovered_shift_mean_ms'], 1)} ms | "
            f"{_format_number(item['recovery_mae_ms'], 1)} ms |"
        )
    recovery = response["recovery"]
    lines.extend([
        "",
        "## Recovery summary",
        "",
        f"- Mean patient-level slope: "
        f"{_format_number(recovery['patient_slope_mean'])}; 95% CI "
        f"[{_format_number(recovery['patient_slope_ci95'][0])}, "
        f"{_format_number(recovery['patient_slope_ci95'][1])}].",
        f"- Mean patient-level Spearman: "
        f"{_format_number(recovery['patient_spearman_mean'])}.",
        f"- Patient shift recovery MAE: "
        f"{_format_number(recovery['patient_recovery_mae_ms'], 1)} ms.",
        f"- Zero-shift minimum rate: "
        f"{_format_number(recovery['zero_shift_is_minimum_rate'])}.",
        "",
        "## Interpretation boundary",
        "",
        "This experiment tests temporal geometry and pairwise shift sensitivity. "
        "It does not show that the ECG-only delay head observes PPG, recover "
        "physical PAT/PTT, or perform causal discovery.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_plots(output_dir: Path, summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plots] matplotlib unavailable; skipping")
        return

    items = sorted(
        summary["response"]["by_injected_shift"].items(),
        key=lambda pair: float(pair[0]),
    )
    shifts = np.asarray([float(key) for key, _ in items])
    loss = np.asarray([item["uncompensated_loss_mean"] for _, item in items])
    recovered = np.asarray([item["recovered_shift_mean_ms"] for _, item in items])
    benefit = np.asarray([
        item["oracle_compensation_benefit_mean"] for _, item in items
    ])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(shifts, loss, "o-", color="#1D3557", label="uncompensated")
    ax.set_xlabel("Injected PPG shift (ms)")
    ax.set_ylabel("Alignment cosine distance")
    ax.set_title("Transport sensitivity to PPG time shift")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "shift_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(shifts, recovered, "o-", color="#2A9D8F", label="loss-profile estimate")
    ax.plot(shifts, shifts, "--", color="#B23A48", label="ideal")
    ax.set_xlabel("Injected PPG shift (ms)")
    ax.set_ylabel("Recovered compensation (ms)")
    ax.set_title("Injected-shift recovery")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "shift_recovery.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#457B9D" if value >= 0 else "#E76F51" for value in benefit]
    ax.bar(shifts.astype(str), benefit, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("Injected PPG shift (ms)")
    ax.set_ylabel("Uncompensated - compensated loss")
    ax.set_title("Known-shift Transport compensation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "shift_compensation.png", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a validation-only PPG time-shift Transport intervention"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data_dir", default="/root/ppgchd/ppgchd/data_updated"
    )
    parser.add_argument(
        "--split", default="splits/multidisease_taskaware_downstream.json"
    )
    parser.add_argument("--role", choices=("train", "val", "test"), default="val")
    parser.add_argument("--allow_test", action="store_true")
    parser.add_argument(
        "--shift_domain",
        choices=("teacher_tokens", "waveform"),
        default="teacher_tokens",
    )
    parser.add_argument(
        "--shifts_ms",
        type=float,
        nargs="+",
        default=[-320.0, -160.0, -80.0, 0.0, 80.0, 160.0, 320.0],
    )
    parser.add_argument(
        "--compensation_grid_ms",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--output_dir", default="outputs_transport_time_shift")
    parser.add_argument("--max_segments", type=int, default=512)
    parser.add_argument("--max_segments_per_patient", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--boundary_margin_tokens", type=int, default=1)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_shift_grid(shifts: Sequence[float], name: str) -> List[float]:
    values = sorted({float(value) for value in shifts})
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")
    return values


def configure_reproducibility(seed: int):
    """Configure the current downstream seeding API for deterministic analysis."""
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def main():
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap_iterations must be at least 100")
    shifts_ms = _validate_shift_grid(args.shifts_ms, "--shifts_ms")
    if not any(abs(value) < 1e-8 for value in shifts_ms):
        raise ValueError("--shifts_ms must include 0")
    compensation_ms = _validate_shift_grid(
        args.compensation_grid_ms or shifts_ms,
        "--compensation_grid_ms",
    )
    for shift in shifts_ms:
        if min(abs(candidate - shift) for candidate in compensation_ms) > 1e-6:
            raise ValueError(
                "Compensation grid must include every injected shift; missing "
                f"{shift} ms"
            )

    configure_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[Runtime] device={device} domain={args.shift_domain} "
        f"shifts_ms={shifts_ms}"
    )

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
    loader = DataLoader(
        NamedMultiDiseaseDataset(base_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    model, model_config, checkpoint = _load_model(args.checkpoint, device)
    sample_rate = float(model_config.model.phase2_sample_rate_hz)
    token_ms = float(model.phase2_token_ms)
    candidate_tokens = [value / token_ms for value in compensation_ms]
    segment_rows = []
    profile_rows = []

    for batch_index, (signals, _, uids, filenames) in enumerate(loader):
        if signals.size(1) < 2:
            raise ValueError("Dual-channel analysis requires PPG and ECG")
        ppg = signals[:, 0:1].to(device, non_blocking=True)
        ecg = signals[:, 1:2].to(device, non_blocking=True)
        prediction, state = _encode_ecg_transport(model, ecg)
        length = prediction.size(1)
        baseline_teacher = _encode_ppg_teacher(model, ppg, length)
        source_valid = state["valid_rows"]
        expected_delay_ms = (
            state["expected_delay"] * token_ms
        )
        base_delay = (
            expected_delay_ms * source_valid.float()
        ).sum(dim=1) / source_valid.float().sum(dim=1).clamp_min(1.0)

        for injected_ms in shifts_ms:
            injected_tokens = injected_ms / token_ms
            if args.shift_domain == "teacher_tokens":
                shifted_teacher, teacher_valid = shift_sequence_non_circular(
                    baseline_teacher, injected_tokens
                )
                shift_samples = int(round(injected_ms * sample_rate / 1000.0))
            else:
                shift_samples = int(round(injected_ms * sample_rate / 1000.0))
                shifted_ppg, waveform_valid = shift_waveform_non_circular(
                    ppg, shift_samples
                )
                shifted_teacher = _encode_ppg_teacher(model, shifted_ppg, length)
                teacher_valid = waveform_mask_to_token_mask(
                    waveform_valid,
                    length,
                    args.boundary_margin_tokens,
                )

            profile = evaluate_compensation_profile(
                prediction,
                shifted_teacher,
                state["forward_transport"],
                source_valid,
                teacher_valid,
                candidate_tokens,
            )
            losses = profile["losses"]
            recovered_ms, minimum_losses = _best_candidate(
                losses, compensation_ms
            )
            zero_index = min(
                range(len(compensation_ms)),
                key=lambda i: abs(compensation_ms[i]),
            )
            oracle_index = min(
                range(len(compensation_ms)),
                key=lambda i: abs(compensation_ms[i] - injected_ms),
            )
            cpu_losses = losses.detach().cpu()
            cpu_valid = profile["valid_fractions"].detach().cpu()
            cpu_overlap = profile["overlap_masses"].detach().cpu()
            base_delay_cpu = base_delay.detach().cpu()

            for index, (uid, filename) in enumerate(zip(uids, filenames)):
                uncompensated = _safe_float(cpu_losses[index, zero_index])
                oracle = _safe_float(cpu_losses[index, oracle_index])
                recovered = recovered_ms[index]
                row = {
                    "uid": str(uid),
                    "file": str(filename),
                    "shift_domain": args.shift_domain,
                    "injected_shift_ms": injected_ms,
                    "injected_shift_samples": shift_samples,
                    "injected_shift_tokens": injected_tokens,
                    "delay_head_mean_ms": _safe_float(base_delay_cpu[index]),
                    "uncompensated_loss": uncompensated,
                    "oracle_compensated_loss": oracle,
                    "oracle_compensation_ms": compensation_ms[oracle_index],
                    "oracle_compensation_benefit": (
                        uncompensated - oracle
                        if math.isfinite(uncompensated) and math.isfinite(oracle)
                        else float("nan")
                    ),
                    "minimum_profile_loss": minimum_losses[index],
                    "recovered_shift_ms": recovered,
                    "recovery_abs_error_ms": (
                        abs(recovered - injected_ms)
                        if math.isfinite(recovered)
                        else float("nan")
                    ),
                    "uncompensated_valid_fraction": _safe_float(
                        cpu_valid[index, zero_index]
                    ),
                    "oracle_valid_fraction": _safe_float(
                        cpu_valid[index, oracle_index]
                    ),
                    "uncompensated_overlap_mass": _safe_float(
                        cpu_overlap[index, zero_index]
                    ),
                    "oracle_overlap_mass": _safe_float(
                        cpu_overlap[index, oracle_index]
                    ),
                }
                segment_rows.append(row)
                for candidate_index, candidate_ms in enumerate(compensation_ms):
                    profile_rows.append({
                        "uid": str(uid),
                        "file": str(filename),
                        "shift_domain": args.shift_domain,
                        "injected_shift_ms": injected_ms,
                        "candidate_compensation_ms": candidate_ms,
                        "loss": _safe_float(
                            cpu_losses[index, candidate_index]
                        ),
                        "valid_fraction": _safe_float(
                            cpu_valid[index, candidate_index]
                        ),
                        "overlap_mass": _safe_float(
                            cpu_overlap[index, candidate_index]
                        ),
                    })
        if (batch_index + 1) % 10 == 0:
            print(
                f"[TimeShift] batches={batch_index + 1}/{len(loader)} "
                f"segment_shift_rows={len(segment_rows)}"
            )

    _attach_zero_shift_deltas(segment_rows)
    patient_rows = _aggregate_patient_shift_rows(segment_rows)
    response = _patient_response_statistics(
        patient_rows,
        shifts_ms,
        args.bootstrap_iterations,
        args.seed,
    )
    summary = {
        "protocol": {
            "analysis_unit": "patient",
            "shift_domain": args.shift_domain,
            "role": args.role,
            "test_set_sealed": args.role != "test",
            "segments": len({row["file"] for row in segment_rows}),
            "patients": len({row["uid"] for row in segment_rows}),
            "sample_rate_hz": sample_rate,
            "token_ms": token_ms,
            "injected_shifts_ms": shifts_ms,
            "compensation_grid_ms": compensation_ms,
            "boundary_handling": "non-circular zero fill with valid-region mask",
            "delay_policy_conditioning": "ECG tokens only",
            "recovered_shift_definition": (
                "argmin candidate compensation under pairwise alignment loss"
            ),
        },
        "response": response,
        "inputs": {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "checkpoint_seed": checkpoint.get("seed"),
            "split": os.path.abspath(args.split),
            "split_sha256": _sha256_file(args.split),
            "data_dir": os.path.abspath(args.data_dir),
            "seed": args.seed,
            "git_sha": _git_sha(),
        },
        "claim_boundary": (
            "Tests temporal geometry and shift sensitivity; does not test "
            "PPG-conditioned delay-head adaptation, PAT/PTT recovery, or "
            "causal discovery."
        ),
    }

    _write_csv(output_dir / "segment_shift_metrics.csv", segment_rows)
    _write_csv(output_dir / "segment_shift_profiles.csv", profile_rows)
    _write_csv(output_dir / "patient_shift_metrics.csv", patient_rows)
    with (output_dir / "transport_time_shift_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    _write_report(output_dir / "transport_time_shift_report.md", summary)
    _make_plots(output_dir, summary)
    print(
        "[Complete] Transport time-shift intervention saved to "
        f"{output_dir.resolve()} | test_set_sealed={args.role != 'test'}"
    )


if __name__ == "__main__":
    main()
