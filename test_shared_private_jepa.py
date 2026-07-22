"""Regression tests for Priority-2 Shared-Private Phase 2 JEPA."""

import types
import unittest

import torch

from models.jepa import JEPA, SharedPrivateTokenProjector
from train_downstream import _select_pretrained_encoder_state
from train_pretrain import (
    _checkpoint_uses_shared_private,
    _encoder_checkpoint_payload,
    _initialize_shared_private_from_phase2,
    _representation_is_healthy,
)


class SharedPrivateJEPATests(unittest.TestCase):
    @staticmethod
    def _tiny_model(enabled: bool = True) -> JEPA:
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
            phase2_sample_rate_hz=100.0,
            phase2_min_delay_ms=40.0,
            phase2_max_delay_ms=160.0,
            phase2_delay_prior_ms=80.0,
            phase2_delay_head_hidden=12,
            phase2_transport_temperature=0.5,
            phase2_shared_private_enabled=enabled,
            phase2_private_dim=6,
            phase2_shared_private_hidden=12,
            phase2_private_loss_weight=0.5,
            phase2_orthogonality_weight=0.05,
        )

    def test_shared_adapter_starts_as_identity(self):
        projector = SharedPrivateTokenProjector(
            dim=8, private_dim=6, hidden_dim=12
        )
        tokens = torch.randn(3, 7, 8)
        shared, private = projector(tokens)

        self.assertTrue(torch.equal(shared, tokens))
        self.assertEqual(tuple(private.shape), (3, 7, 6))

    def test_private_objective_is_finite_and_backpropagates(self):
        torch.manual_seed(23)
        model = self._tiny_model()
        model.train()
        model.set_phase2_progress(1.0)
        model.set_shared_private_progress(1.0)

        loss, info, components = model.compute_loss(
            torch.randn(4, 1, 64),
            torch.randn(4, 1, 64),
            return_components=True,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for key in (
            "private_reconstruction",
            "ecg_private_reconstruction",
            "ppg_private_reconstruction",
            "shared_private_orthogonality",
            "ecg_shared_std",
            "ppg_private_std",
        ):
            self.assertIn(key, info)
            self.assertTrue(torch.isfinite(torch.tensor(info[key])))
        for key in (
            "private_reconstruction",
            "shared_private_orthogonality",
            "total",
        ):
            self.assertIn(key, components)
            self.assertTrue(torch.isfinite(components[key]))

        for module in (
            model.context_encoder,
            model.ppg_encoder,
            model.ecg_shared_private,
            model.ppg_shared_private,
            model.ecg_private_predictor,
            model.ppg_private_predictor,
        ):
            gradient = sum(
                parameter.grad.abs().sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

        for teacher in (
            model.context_teacher,
            model.target_encoder,
            model.ecg_teacher_shared_private,
            model.ppg_teacher_shared_private,
        ):
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_causal_delay_head_receives_shared_not_base_tokens(self):
        torch.manual_seed(29)
        model = self._tiny_model()
        model.set_phase2_progress(1.0)
        model.set_shared_private_progress(1.0)
        captured = {}

        def capture_base(_module, _inputs, output):
            captured["base"] = output.detach().clone()

        def capture_delay(_module, inputs):
            captured["delay"] = inputs[0].detach().clone()

        original_forward = model.ecg_shared_private.forward

        def shifted_shared(module, tokens):
            _, private = original_forward(tokens)
            return tokens + 3.0, private

        model.ecg_shared_private.forward = types.MethodType(
            shifted_shared, model.ecg_shared_private
        )
        base_hook = model.ecg_token_proj.register_forward_hook(capture_base)
        delay_hook = model.phase2_delay_head.register_forward_pre_hook(
            capture_delay
        )
        try:
            model.compute_loss(
                torch.randn(3, 1, 64), torch.randn(3, 1, 64)
            )
        finally:
            base_hook.remove()
            delay_hook.remove()

        self.assertTrue(torch.allclose(captured["delay"], captured["base"] + 3.0))

    def test_orthogonality_penalizes_correlated_views(self):
        shared = torch.tensor(
            [[[-1.0, -1.0]], [[-0.5, 0.5]], [[0.5, -0.5]], [[1.0, 1.0]]]
        )
        correlated = shared.clone()
        constant = torch.ones_like(shared)

        correlated_loss = JEPA._shared_private_orthogonality(
            shared, correlated
        )
        constant_loss = JEPA._shared_private_orthogonality(shared, constant)
        self.assertGreater(correlated_loss.item(), constant_loss.item())

    def test_ema_updates_private_teachers(self):
        model = self._tiny_model()
        with torch.no_grad():
            next(model.ecg_shared_private.parameters()).fill_(2.0)
            next(model.ecg_teacher_shared_private.parameters()).zero_()
        model.update_target_encoder(momentum=0.0)

        self.assertTrue(torch.equal(
            next(model.ecg_shared_private.parameters()),
            next(model.ecg_teacher_shared_private.parameters()),
        ))
        self.assertFalse(model.ecg_teacher_shared_private.training)

    def test_legacy_phase2_checkpoint_initializes_only_new_modules(self):
        legacy = self._tiny_model(enabled=False)
        model = self._tiny_model(enabled=True)
        checkpoint = {
            "pretrain_phase": 2,
            "model_state_dict": legacy.state_dict(),
            **_encoder_checkpoint_payload(legacy),
        }

        missing = _initialize_shared_private_from_phase2(model, checkpoint)
        self.assertTrue(missing)
        self.assertTrue(all(
            key.startswith((
                "ecg_shared_private.",
                "ppg_shared_private.",
                "ecg_teacher_shared_private.",
                "ppg_teacher_shared_private.",
                "ecg_private_predictor.",
                "ppg_private_predictor.",
            ))
            for key in missing
        ))
        self.assertFalse(_checkpoint_uses_shared_private(checkpoint))

        new_checkpoint = {
            "pretrain_phase": 2,
            "model_state_dict": model.state_dict(),
        }
        self.assertTrue(_checkpoint_uses_shared_private(new_checkpoint))

    def test_shared_private_checkpoint_remains_downstream_compatible(self):
        model = self._tiny_model()
        payload = _encoder_checkpoint_payload(model)
        checkpoint = {"pretrain_phase": 2, **payload}

        ppg_state, ppg_key = _select_pretrained_encoder_state(
            checkpoint, "target"
        )
        ecg_state, ecg_key = _select_pretrained_encoder_state(
            checkpoint, "context"
        )
        self.assertEqual(ppg_key, "ppg_encoder")
        self.assertEqual(ecg_key, "context_encoder")
        self.assertEqual(ppg_state.keys(), payload["ppg_encoder"].keys())
        self.assertEqual(ecg_state.keys(), payload["context_encoder"].keys())

    def test_checkpoint_health_rejects_collapsed_private_view(self):
        metrics = {
            "total_loss": 0.1,
            "context_std": 0.2,
            "target_std": 0.2,
            "context_collapsed_fraction": 0.0,
            "target_collapsed_fraction": 0.0,
            "ecg_private_std": 0.2,
            "ppg_private_std": 0.001,
            "ecg_private_collapsed_fraction": 0.0,
            "ppg_private_collapsed_fraction": 1.0,
        }
        self.assertFalse(_representation_is_healthy(metrics))
        metrics["ppg_private_std"] = 0.2
        metrics["ppg_private_collapsed_fraction"] = 0.0
        self.assertTrue(_representation_is_healthy(metrics))


if __name__ == "__main__":
    unittest.main()
