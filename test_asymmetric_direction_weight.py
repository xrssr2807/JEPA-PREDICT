"""Regression tests for the Phase 2 asymmetric bidirectional objective."""

import unittest

import torch

from config import Config
from models.jepa import JEPA
from train_pretrain import _phase_checkpoint_metadata


class AsymmetricDirectionWeightTests(unittest.TestCase):
    @staticmethod
    def _model(alpha: float = 1.0) -> JEPA:
        return JEPA(
            cnn_channels=(8, 16),
            cnn_kernel_sizes=(3, 3),
            cnn_strides=(2, 2),
            transformer_layers=1,
            transformer_dim=16,
            transformer_heads=4,
            transformer_ff_dim=32,
            transformer_dropout=0.0,
            max_seq_len=32,
            embedding_dim=8,
            predictor_hidden=16,
            latent_dim=4,
            num_latent_samples=2,
            pretrain_phase=2,
            phase1_mask_ratio=0.5,
            phase1_mask_block_tokens=3,
            phase2_transport_mode="physio_v2",
            phase2_sample_rate_hz=100.0,
            phase2_min_delay_ms=40.0,
            phase2_max_delay_ms=160.0,
            phase2_delay_prior_ms=80.0,
            phase2_delay_head_hidden=12,
            phase2_transport_temperature=0.5,
            phase2_v2_transport_dim=6,
            phase2_v2_sinkhorn_iters=3,
            phase2_reverse_loss_weight=alpha,
        )

    def test_alpha_zero_is_forward_only(self):
        model = self._model(alpha=0.0)
        forward = torch.tensor(2.0)
        reverse = torch.tensor(10.0)
        self.assertEqual(
            model._combine_phase2_direction_losses(forward, reverse).item(),
            2.0,
        )

    def test_alpha_one_reproduces_symmetric_average(self):
        model = self._model(alpha=1.0)
        forward = torch.tensor(2.0)
        reverse = torch.tensor(10.0)
        self.assertEqual(
            model._combine_phase2_direction_losses(forward, reverse).item(),
            6.0,
        )

    def test_intermediate_alpha_is_scale_normalized(self):
        model = self._model(alpha=0.25)
        forward = torch.tensor(2.0)
        reverse = torch.tensor(10.0)
        expected = (2.0 + 0.25 * 10.0) / 1.25
        self.assertAlmostEqual(
            model._combine_phase2_direction_losses(forward, reverse).item(),
            expected,
            places=6,
        )

    def test_invalid_alpha_is_rejected(self):
        for alpha in (-0.01, 1.01):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "reverse_loss_weight"):
                    self._model(alpha=alpha)

    def test_full_loss_reports_directional_components(self):
        torch.manual_seed(37)
        model = self._model(alpha=0.25)
        model.set_phase2_progress(1.0)
        loss, info, components = model.compute_loss(
            torch.randn(3, 1, 64),
            torch.randn(3, 1, 64),
            return_components=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(info["reverse_loss_weight"], 0.25)
        for key in (
            "direct_ecg_to_ppg_token",
            "direct_ppg_to_ecg_token",
            "transport_ecg_to_ppg_token",
            "transport_ppg_to_ecg_token",
        ):
            self.assertIn(key, components)
            self.assertTrue(torch.isfinite(components[key]))

    def test_checkpoint_metadata_records_alpha(self):
        model = self._model(alpha=0.25)
        config = Config()
        config.model.pretrain_phase = 2
        config.model.phase2_transport_mode = "physio_v2"
        config.model.phase2_reverse_loss_weight = 0.25
        metadata = _phase_checkpoint_metadata(model, config)["phase2_config"]
        self.assertEqual(metadata["reverse_loss_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
