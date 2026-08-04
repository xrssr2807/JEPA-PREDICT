"""Regression tests for paired-content physiological Transport v2."""

import unittest

import torch

from models.jepa import JEPA
from train_pretrain import _initialize_shared_private_from_phase2


class PhysiologicalTransportV2Tests(unittest.TestCase):
    @staticmethod
    def _model(**overrides) -> JEPA:
        kwargs = dict(
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
            phase2_v2_sinkhorn_iters=5,
        )
        kwargs.update(overrides)
        return JEPA(**kwargs)

    def test_transport_requires_the_ppg_endpoint(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "paired PPG"):
            model._build_phase2_transport(torch.randn(2, 12, 8))

    def test_plan_is_finite_causal_and_banded(self):
        torch.manual_seed(5)
        model = self._model()
        state = model._build_phase2_transport(
            torch.randn(2, 12, 8), torch.randn(2, 12, 8)
        )
        plan = state["transport"]
        self.assertTrue(torch.isfinite(plan).all())
        self.assertTrue((plan >= 0).all())
        self.assertLessEqual(plan.sum(dim=-1).max().item(), 1.0 + 1e-5)

        allowed = torch.zeros(12, 12, dtype=torch.bool)
        for offset in model.phase2_delay_offsets.tolist():
            source = torch.arange(12 - offset)
            allowed[source, source + offset] = True
        self.assertEqual(plan[:, ~allowed].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(state["sinkhorn_row_error"]))
        self.assertTrue(torch.isfinite(state["sinkhorn_column_error"]))

    def test_plan_depends_on_both_modalities(self):
        torch.manual_seed(11)
        model = self._model()
        ecg = torch.randn(2, 12, 8)
        ppg = torch.randn(2, 12, 8)
        first = model._build_phase2_transport(ecg, ppg)[
            "forward_transport"
        ]
        second = model._build_phase2_transport(ecg, torch.flip(ppg, (1,)))[
            "forward_transport"
        ]
        self.assertGreater((first - second).abs().max().item(), 1e-5)

    def test_full_loss_and_cross_modal_head_have_finite_gradients(self):
        torch.manual_seed(19)
        model = self._model()
        model.train()
        model.set_phase2_progress(1.0)
        loss, info = model.compute_loss(
            torch.randn(3, 1, 64), torch.randn(3, 1, 64)
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.phase2_physio_transport.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        for key in (
            "counterfactual_loss",
            "counterfactual_accuracy",
            "sinkhorn_row_error",
            "sinkhorn_column_error",
        ):
            self.assertIn(key, info)
            self.assertTrue(torch.isfinite(torch.tensor(info[key])))

    def test_optional_pat_supervision_is_confidence_weighted(self):
        torch.manual_seed(23)
        model = self._model(phase2_pat_weak_weight=0.2)
        model.set_phase2_progress(1.0)
        loss, info = model.compute_loss(
            torch.randn(3, 1, 64),
            torch.randn(3, 1, 64),
            pat_target_ms=torch.tensor([80.0, float("nan"), 120.0]),
            pat_confidence=torch.tensor([1.0, 1.0, 0.5]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(info["pat_weak_loss"], 0.0)
        self.assertAlmostEqual(info["pat_valid_fraction"], 2.0 / 3.0, places=5)

    def test_legacy_full_checkpoint_can_initialize_v2(self):
        legacy = self._model(phase2_transport_mode="full")
        target = self._model()
        checkpoint = {
            "pretrain_phase": 2,
            "phase2_config": {
                "transport_enabled": True,
                "transport_mode": "full",
                "shared_private_enabled": False,
            },
            "model_state_dict": legacy.state_dict(),
        }
        missing = _initialize_shared_private_from_phase2(target, checkpoint)
        self.assertTrue(missing)
        self.assertTrue(all(
            key.startswith("phase2_physio_transport.") for key in missing
        ))


if __name__ == "__main__":
    unittest.main()
