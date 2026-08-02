import unittest
from unittest.mock import patch

import torch

from analyze_transport_time_shift import (
    configure_reproducibility,
    evaluate_compensation_profile,
    evaluate_reference_shift_profile,
    shift_sequence_non_circular,
    shift_transport_columns,
    shift_waveform_non_circular,
)


class TransportTimeShiftTests(unittest.TestCase):
    @patch("analyze_transport_time_shift.seed_everything")
    def test_reproducibility_uses_current_seed_api(self, seed_mock):
        configure_reproducibility(3407)
        seed_mock.assert_called_once_with(3407)

    def test_waveform_shift_has_no_wraparound(self):
        waveform = torch.arange(1, 7, dtype=torch.float32).view(1, 1, -1)

        delayed, delayed_valid = shift_waveform_non_circular(waveform, 2)
        self.assertTrue(torch.equal(
            delayed,
            torch.tensor([[[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]]]),
        ))
        self.assertTrue(torch.equal(
            delayed_valid,
            torch.tensor([[False, False, True, True, True, True]]),
        ))

        advanced, advanced_valid = shift_waveform_non_circular(waveform, -2)
        self.assertTrue(torch.equal(
            advanced,
            torch.tensor([[[3.0, 4.0, 5.0, 6.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            advanced_valid,
            torch.tensor([[True, True, True, True, False, False]]),
        ))

    def test_fractional_sequence_shift_is_non_circular(self):
        sequence = torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])
        shifted, valid = shift_sequence_non_circular(sequence, 0.5)

        self.assertTrue(torch.allclose(
            shifted[:, 1:, 0],
            torch.tensor([[1.0, 3.0, 5.0]]),
        ))
        self.assertFalse(bool(valid[0, 0]))
        self.assertTrue(bool(valid[0, 1:].all()))
        self.assertEqual(float(shifted[0, 0, 0]), 0.0)

    def test_transport_column_shift_preserves_direction_without_wrap(self):
        plan = torch.eye(5, dtype=torch.float32).unsqueeze(0)
        shifted, row_mass = shift_transport_columns(plan, 1.0)

        expected = torch.zeros_like(plan)
        expected[0, 0, 1] = 1.0
        expected[0, 1, 2] = 1.0
        expected[0, 2, 3] = 1.0
        expected[0, 3, 4] = 1.0
        self.assertTrue(torch.allclose(shifted, expected))
        self.assertTrue(torch.allclose(
            row_mass,
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]]),
        ))

    def test_profile_recovers_known_positive_shift(self):
        length = 7
        teacher = torch.eye(length, dtype=torch.float32).unsqueeze(0)
        prediction = teacher.clone()
        plan = torch.eye(length, dtype=torch.float32).unsqueeze(0)
        source_valid = torch.ones((1, length), dtype=torch.bool)

        shifted_teacher, teacher_valid = shift_sequence_non_circular(
            teacher, 1.0
        )
        profile = evaluate_compensation_profile(
            prediction,
            shifted_teacher,
            plan,
            source_valid,
            teacher_valid,
            candidate_shifts_tokens=[-1.0, 0.0, 1.0],
        )

        losses = profile["losses"][0]
        finite_losses = torch.where(
            torch.isfinite(losses),
            losses,
            torch.full_like(losses, float("inf")),
        )
        best = int(torch.argmin(finite_losses).item())
        self.assertEqual(profile["candidate_shifts_tokens"][best], 1.0)
        self.assertLess(float(losses[best]), 1e-6)
        self.assertGreater(float(losses[1]), float(losses[best]) + 0.5)

    def test_reference_profile_recovers_shift_without_transport(self):
        teacher = torch.eye(7, dtype=torch.float32).unsqueeze(0)
        shifted, valid = shift_sequence_non_circular(teacher, -1.0)
        profile = evaluate_reference_shift_profile(
            teacher,
            shifted,
            valid,
            candidate_shifts_tokens=[-1.0, 0.0, 1.0],
        )
        losses = profile["losses"][0]
        best = int(torch.nan_to_num(losses, nan=float("inf")).argmin())
        self.assertEqual(profile["candidate_shifts_tokens"][best], -1.0)
        self.assertLess(float(losses[best]), 1e-6)


if __name__ == "__main__":
    unittest.main()
