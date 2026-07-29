import math
import unittest

import numpy as np
import torch

from analyze_transport_interpretability import (
    _limit_segments_per_patient,
    _shifted_targets,
    detect_r_peaks,
    estimate_pat_proxy,
    spearman_correlation,
    transport_batch_diagnostics,
)
from models.jepa import JEPA


def _small_phase2_model():
    model = JEPA(
        cnn_channels=(16, 32, 64, 64),
        cnn_kernel_sizes=(7, 5, 5, 3),
        cnn_strides=(2, 2, 2, 2),
        transformer_layers=1,
        transformer_dim=64,
        transformer_heads=4,
        transformer_ff_dim=128,
        transformer_dropout=0.0,
        max_seq_len=200,
        embedding_dim=32,
        predictor_hidden=32,
        pretrain_phase=2,
        phase2_sample_rate_hz=100.0,
        phase2_min_delay_ms=80.0,
        phase2_max_delay_ms=800.0,
    )
    model.eval()
    return model


class TransportInterpretabilityTests(unittest.TestCase):
    def test_detect_r_peaks_and_pat_proxy(self):
        sample_rate = 100.0
        length = 1000
        ecg = np.zeros(length, dtype=np.float64)
        ppg = np.zeros(length, dtype=np.float64)
        r_peaks = np.arange(100, 900, 100)
        delay = 25
        for peak in r_peaks:
            ecg[peak] = 5.0
            foot = peak + delay
            ppg[foot:foot + 8] = np.linspace(0.0, 1.0, 8)
            ppg[foot + 8:foot + 25] = np.linspace(1.0, 0.0, 17)

        detected = detect_r_peaks(ecg, sample_rate)
        self.assertGreaterEqual(detected.size, 6)
        result = estimate_pat_proxy(ecg, ppg, sample_rate, 80.0, 800.0)
        self.assertGreaterEqual(result["beat_count"], 6)
        self.assertLessEqual(
            abs(result["pat_foot_median_ms"] - 250.0), 30.0
        )
        self.assertTrue(result["quality_pass"])

    def test_r_peak_detector_rejects_broad_t_waves(self):
        sample_rate = 100.0
        length = 1000
        samples = np.arange(length)
        ecg = np.zeros(length, dtype=np.float64)
        expected = np.arange(100, 900, 100)
        for peak in expected:
            ecg += 5.0 * np.exp(-0.5 * ((samples - peak) / 1.5) ** 2)
            t_wave = peak + 35
            ecg += 2.5 * np.exp(-0.5 * ((samples - t_wave) / 10.0) ** 2)

        detected = detect_r_peaks(ecg, sample_rate)
        self.assertEqual(detected.size, expected.size)
        for actual, target in zip(detected, expected):
            self.assertLessEqual(abs(int(actual) - int(target)), 3)

    def test_shifted_targets_respect_direction(self):
        tokens = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        offsets = torch.tensor([1, 2])
        probabilities = torch.zeros(1, 8, 2)
        probabilities[..., 0] = 1.0
        forward, forward_valid = _shifted_targets(
            tokens, offsets, probabilities, direction=1
        )
        backward, backward_valid = _shifted_targets(
            tokens, offsets, probabilities, direction=-1
        )
        self.assertTrue(
            torch.allclose(forward[0, :7, 0], tokens[0, 1:, 0])
        )
        self.assertTrue(
            torch.allclose(backward[0, 1:, 0], tokens[0, :-1, 0])
        )
        self.assertTrue(forward_valid[0, 0])
        self.assertTrue(backward_valid[0, 1])

    def test_transport_batch_diagnostics_are_finite(self):
        model = _small_phase2_model()
        ecg = torch.randn(3, 1, 1000)
        ppg = torch.randn(3, 1, 1000)
        diagnostics = transport_batch_diagnostics(model, ecg, ppg)
        for name in (
            "dynamic_causal",
            "segment_static_delay",
            "token_shuffled_delay",
            "cross_patient_delay_policy",
            "fixed_prior",
            "zero_delay",
            "negative_delay",
            "reversed_ppg",
            "shuffled_pair",
            "delay_mean_ms",
            "delay_std_ms",
            "monotonic_violation_rate",
            "matched_mass",
        ):
            self.assertEqual(diagnostics[name].shape, (3,))
            self.assertTrue(torch.isfinite(diagnostics[name]).all())
        self.assertTrue((diagnostics["delay_mean_ms"] > 0).all())
        self.assertTrue(
            (diagnostics["monotonic_violation_rate"] >= 0).all()
        )
        self.assertTrue(
            (diagnostics["monotonic_violation_rate"] <= 1).all()
        )

    def test_spearman_correlation(self):
        self.assertTrue(math.isclose(
            spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]),
            1.0,
        ))
        self.assertTrue(math.isclose(
            spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]),
            -1.0,
        ))

    def test_patient_segment_cap_is_deterministic_and_balanced(self):
        files = [
            "val_a_0.pkl",
            "val_a_1.pkl",
            "val_a_2.pkl",
            "val_b_0.pkl",
            "val_b_1.pkl",
            "val_c_0.pkl",
        ]
        first = _limit_segments_per_patient(files, 2, seed=42)
        second = _limit_segments_per_patient(files, 2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(
            sum(name.startswith("val_a_") for name in first), 2
        )
        self.assertEqual(
            sum(name.startswith("val_b_") for name in first), 2
        )
        self.assertEqual(
            sum(name.startswith("val_c_") for name in first), 1
        )


if __name__ == "__main__":
    unittest.main()
