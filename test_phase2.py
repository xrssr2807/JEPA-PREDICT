"""Regression tests for Phase 2 causal monotonic transport JEPA."""
import unittest

import torch

from models.jepa import JEPA
from train_downstream import _select_pretrained_encoder_state
from train_pretrain import (
    _checkpoint_is_eligible,
    _early_stopping_step,
    _encoder_checkpoint_payload,
    phase2_transport_progress,
)


class Phase2JEPATests(unittest.TestCase):
    @staticmethod
    def _tiny_model(transport_enabled=True):
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
            phase1_bidirectional=True,
            phase2_transport_enabled=transport_enabled,
            phase2_sample_rate_hz=100.0,
            phase2_min_delay_ms=40.0,
            phase2_max_delay_ms=160.0,
            phase2_delay_prior_ms=80.0,
            phase2_delay_head_hidden=12,
            phase2_transport_temperature=0.5,
        )

    def test_transport_schedule_warms_up_then_ramps(self):
        self.assertEqual(phase2_transport_progress(0, 2, 4), 0.0)
        self.assertEqual(phase2_transport_progress(1, 2, 4), 0.0)
        self.assertAlmostEqual(phase2_transport_progress(2, 2, 4), 0.25)
        self.assertAlmostEqual(phase2_transport_progress(5, 2, 4), 1.0)
        self.assertAlmostEqual(phase2_transport_progress(20, 2, 4), 1.0)

    def test_best_checkpoint_waits_for_full_transport(self):
        healthy = {
            "context_std": 0.2,
            "target_std": 0.2,
            "context_collapsed_fraction": 0.0,
            "target_collapsed_fraction": 0.0,
        }
        self.assertFalse(_checkpoint_is_eligible(healthy, 2, 0.95))
        self.assertTrue(_checkpoint_is_eligible(healthy, 2, 1.0))
        self.assertTrue(_checkpoint_is_eligible(healthy, 1, 0.0))
        self.assertTrue(_checkpoint_is_eligible(
            healthy,
            2,
            transport_progress=0.0,
            transport_required=False,
        ))

    def test_transport_ablation_uses_direct_loss_only(self):
        model = self._tiny_model(transport_enabled=False)
        model.train()
        model.set_phase2_progress(1.0)
        loss, info, components = model.compute_loss(
            torch.randn(3, 1, 64),
            torch.randn(3, 1, 64),
            return_components=True,
        )

        self.assertEqual(model.phase2_progress, 0.0)
        self.assertFalse(info["phase2_transport_enabled"])
        self.assertEqual(info["transport_token_jepa"], 0.0)
        self.assertTrue(torch.allclose(
            components["token_jepa"],
            components["direct_token_jepa"],
        ))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.phase2_delay_head.parameters()
        ))

    def test_early_stopping_requires_meaningful_validation_decrease(self):
        best, bad_epochs, improved = _early_stopping_step(
            0.0300, 4, 0.02995, 1e-4
        )
        self.assertFalse(improved)
        self.assertEqual(best, 0.0300)
        self.assertEqual(bad_epochs, 5)

        best, bad_epochs, improved = _early_stopping_step(
            best, bad_epochs, 0.0298, 1e-4
        )
        self.assertTrue(improved)
        self.assertEqual(best, 0.0298)
        self.assertEqual(bad_epochs, 0)

    def test_delay_bins_retain_physical_time_scale(self):
        model = self._tiny_model()
        self.assertEqual(model.phase2_token_ms, 40.0)
        self.assertEqual(model.phase2_delay_offsets.tolist(), [1, 2, 3, 4])

    def test_transport_is_causal_banded_and_unbalanced(self):
        model = self._tiny_model()
        tokens = torch.randn(2, 12, 8)
        state = model._build_phase2_transport(tokens)
        transport = state["transport"]

        self.assertEqual(tuple(transport.shape), (2, 12, 12))
        self.assertTrue(torch.all(transport >= 0))
        self.assertTrue(torch.all(transport.sum(dim=-1) <= 1.0 + 1e-6))
        self.assertEqual(torch.tril(transport).abs().max().item(), 0.0)
        self.assertGreater(state["unmatched_probability"][:, -1].min().item(), 0.99)

        forward = state["forward_transport"]
        reverse = state["reverse_transport"]
        self.assertEqual(torch.tril(forward).abs().max().item(), 0.0)
        self.assertEqual(torch.tril(reverse).abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(
            forward.sum(dim=-1)[state["valid_rows"]],
            torch.ones_like(forward.sum(dim=-1)[state["valid_rows"]]),
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            reverse.sum(dim=1)[state["valid_columns"]],
            torch.ones_like(reverse.sum(dim=1)[state["valid_columns"]]),
            atol=1e-6,
        ))

    def test_regularizers_stay_finite_with_one_matchable_row(self):
        model = self._tiny_model()
        state = model._build_phase2_transport(torch.randn(2, 2, 8))
        regularizers = model._phase2_transport_regularizers(state)
        self.assertEqual(regularizers["monotonic_loss"].item(), 0.0)
        self.assertEqual(regularizers["delay_smoothness_loss"].item(), 0.0)
        self.assertTrue(all(torch.isfinite(value) for value in regularizers.values()))

    def test_full_phase2_loss_is_finite_and_backpropagates(self):
        model = self._tiny_model()
        model.train()
        model.set_phase2_progress(1.0)
        torch.manual_seed(19)
        ecg = torch.randn(3, 1, 64)
        ppg = torch.randn(3, 1, 64)

        loss, info = model.compute_loss(ecg, ppg)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for module in (
            model.context_encoder,
            model.ppg_encoder,
            model.phase2_delay_head,
        ):
            gradient = sum(
                parameter.grad.abs().sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.context_teacher.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.target_encoder.parameters()
        ))
        for key in (
            "direct_token_jepa",
            "transport_token_jepa",
            "delay_mean_ms",
            "monotonic",
            "matched_mass",
        ):
            self.assertIn(key, info)
            self.assertTrue(torch.isfinite(torch.tensor(info[key])))

    def test_phase2_can_return_differentiable_components(self):
        model = self._tiny_model()
        model.set_phase2_progress(1.0)
        result = model.compute_loss(
            torch.randn(3, 1, 64),
            torch.randn(3, 1, 64),
            return_components=True,
        )
        self.assertEqual(len(result), 3)
        loss, _, components = result
        self.assertIs(components["total"], loss)
        for key in (
            "direct_token_jepa",
            "transport_token_jepa",
            "token_jepa",
            "delay_prior",
            "monotonic",
            "delay_smoothness",
            "match_mass",
            "variance",
            "covariance",
            "total",
        ):
            self.assertIn(key, components)
            self.assertTrue(torch.is_tensor(components[key]))
            self.assertTrue(torch.isfinite(components[key]))

    def test_variance_regularizer_penalizes_collapsed_embeddings(self):
        collapsed = torch.zeros(8, 16)
        diverse = torch.randn(8, 16)
        collapsed_loss, _ = self._tiny_model()._variance_covariance_regularization(
            (collapsed,), target_std=0.1
        )
        diverse_loss, _ = self._tiny_model()._variance_covariance_regularization(
            (diverse,), target_std=0.1
        )
        self.assertGreater(collapsed_loss.item(), diverse_loss.item())

    def test_low_match_mass_has_finite_amp_scaled_gradients(self):
        """A confident dustbin prediction must not create 1/mass gradients."""
        torch.manual_seed(3)
        model = self._tiny_model()
        model.train()
        model.set_phase2_progress(0.25)
        with torch.no_grad():
            model.phase2_delay_head.output.weight.zero_()
            model.phase2_delay_head.output.bias.zero_()
            model.phase2_delay_head.output.bias[-1] = 5.0

        ecg = torch.randn(3, 1, 64)
        ppg = torch.randn(3, 1, 64)
        with torch.autocast("cpu", dtype=torch.float16):
            loss, info = model.compute_loss(ecg, ppg)
        self.assertLess(info["matched_mass"], 1e-3)
        (loss * 4096.0).backward()

        self.assertTrue(all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ))

    def test_phase2_checkpoint_remains_downstream_compatible(self):
        model = self._tiny_model()
        payload = _encoder_checkpoint_payload(model)
        checkpoint = {"pretrain_phase": 2, **payload}
        state, key = _select_pretrained_encoder_state(checkpoint, "target")

        self.assertEqual(key, "ppg_encoder")
        self.assertEqual(state.keys(), payload["ppg_encoder"].keys())


if __name__ == "__main__":
    unittest.main()
