"""Physical-time-preserving sampling-rate transforms for biosignals."""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample_poly


def _positive_rate(value, fallback: float) -> float:
    try:
        rate = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        rate = float(fallback)
    return rate if np.isfinite(rate) and rate > 0 else float(fallback)


def polyphase_resample(
    signal: np.ndarray,
    source_hz: float,
    target_hz: float,
    axis: int = -1,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """Resample with an anti-aliasing polyphase filter and stable length."""

    source_hz = _positive_rate(source_hz, 1.0)
    target_hz = _positive_rate(target_hz, source_hz)
    ratio = Fraction(target_hz / source_hz).limit_denominator(1000)
    output = resample_poly(
        np.asarray(signal, dtype=np.float32),
        ratio.numerator,
        ratio.denominator,
        axis=axis,
    ).astype(np.float32, copy=False)
    if expected_length is None:
        expected_length = int(round(signal.shape[axis] * target_hz / source_hz))
    current = output.shape[axis]
    if current > expected_length:
        slices = [slice(None)] * output.ndim
        slices[axis] = slice(0, expected_length)
        output = output[tuple(slices)]
    elif current < expected_length:
        padding = [(0, 0)] * output.ndim
        padding[axis] = (0, expected_length - current)
        output = np.pad(output, padding, mode="edge")
    return output.astype(np.float32, copy=False)


def bridge_device_sampling_rate(
    signal: np.ndarray,
    source_hz: float,
    canonical_hz: float,
    device_hz: Optional[float] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, dict]:
    """Map a signal to a canonical physical-time grid.

    When ``device_hz`` is lower than the source rate, the function first
    simulates acquisition at that device rate using anti-aliased downsampling,
    then maps the result to the model's canonical rate. Duration is preserved.
    """

    source_hz = _positive_rate(source_hz, canonical_hz)
    canonical_hz = _positive_rate(canonical_hz, source_hz)
    duration_seconds = signal.shape[axis] / source_hz
    canonical_length = max(1, int(round(duration_seconds * canonical_hz)))
    effective_device_hz = source_hz
    intermediate = np.asarray(signal, dtype=np.float32)
    if device_hz is not None:
        requested = _positive_rate(device_hz, source_hz)
        effective_device_hz = min(requested, source_hz)
        device_length = max(1, int(round(duration_seconds * effective_device_hz)))
        intermediate = polyphase_resample(
            intermediate,
            source_hz,
            effective_device_hz,
            axis=axis,
            expected_length=device_length,
        )
    output = polyphase_resample(
        intermediate,
        effective_device_hz,
        canonical_hz,
        axis=axis,
        expected_length=canonical_length,
    )
    return output, {
        "source_hz": source_hz,
        "device_hz": effective_device_hz,
        "canonical_hz": canonical_hz,
        "duration_seconds": duration_seconds,
        "canonical_length": canonical_length,
    }
