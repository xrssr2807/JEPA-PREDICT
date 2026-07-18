"""Regression tests for Phase 1 dual-online masked-token JEPA."""
import unittest

import torch

from models.jepa import JEPA
from train_downstream import _select_pretrained_encoder_state
from train_pretrain import _encoder_checkpoint_payload


class Phase1JEPATests(unittest.TestCase):
    @staticmethod
    def _tiny_model():
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
            pretrain_phase=1,
            phase1_mask_ratio=0.5,
            phase1_mask_block_tokens=3,
            phase1_bidirectional=True,
        )

    def test_dual_online_and_same_modality_teachers(self):
        model = self._tiny_model()
        model.train()
        self.assertTrue(model.context_encoder.training)
        self.assertTrue(model.ppg_encoder.training)
        self.assertFalse(model.context_teacher.training)
        self.assertFalse(model.target_encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.context_teacher.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.target_encoder.parameters()))

    def test_block_mask_has_exact_ratio_and_visible_context(self):
        model = self._tiny_model()
        torch.manual_seed(7)
        mask = model._make_token_block_mask(4, 16, torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (4, 16))
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.full((4,), 8)))
        self.assertTrue((~mask).any(dim=1).all())
        adjacent = (mask[:, 1:] & mask[:, :-1]).any(dim=1)
        self.assertTrue(adjacent.all())

    def test_both_online_encoders_receive_gradients(self):
        model = self._tiny_model()
        model.train()
        torch.manual_seed(11)
        ecg = torch.randn(3, 1, 64)
        ppg = torch.randn(3, 1, 64)
        loss, info = model.compute_loss(ecg, ppg)
        loss.backward()

        ecg_grad = sum(
            p.grad.abs().sum().item()
            for p in model.context_encoder.parameters() if p.grad is not None
        )
        ppg_grad = sum(
            p.grad.abs().sum().item()
            for p in model.ppg_encoder.parameters() if p.grad is not None
        )
        self.assertGreater(ecg_grad, 0.0)
        self.assertGreater(ppg_grad, 0.0)
        self.assertTrue(all(p.grad is None for p in model.context_teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
        self.assertIn("ecg_to_ppg_token", info)
        self.assertIn("ppg_to_ecg_token", info)
        self.assertAlmostEqual(info["masked_fraction"], 0.5, places=6)

    def test_token_loss_has_dimension_independent_cosine_scale(self):
        target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        prediction = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        mask = torch.tensor([[True, True]])
        loss = self._tiny_model()._masked_token_regression(
            prediction, target, mask
        )
        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_ema_tracks_each_modality_without_cross_overwrite(self):
        model = self._tiny_model()
        with torch.no_grad():
            next(model.context_encoder.parameters()).fill_(1.0)
            next(model.ppg_encoder.parameters()).fill_(2.0)
            next(model.context_teacher.parameters()).zero_()
            next(model.target_encoder.parameters()).zero_()
        model.update_target_encoder(momentum=0.0)
        self.assertTrue(torch.allclose(
            next(model.context_teacher.parameters()),
            next(model.context_encoder.parameters()),
        ))
        self.assertTrue(torch.allclose(
            next(model.target_encoder.parameters()),
            next(model.ppg_encoder.parameters()),
        ))
        self.assertFalse(torch.allclose(
            next(model.target_encoder.parameters()),
            next(model.context_encoder.parameters()),
        ))

    def test_checkpoint_exposes_online_ppg_for_downstream(self):
        payload = _encoder_checkpoint_payload(self._tiny_model())
        self.assertIn("context_encoder", payload)
        self.assertIn("ppg_encoder", payload)
        self.assertIn("context_teacher", payload)
        self.assertIn("target_encoder", payload)

    def test_downstream_target_role_prefers_online_ppg(self):
        checkpoint = {
            "pretrain_phase": 1,
            "ppg_encoder": {"weight": torch.tensor([2.0])},
            "target_encoder": {"weight": torch.tensor([3.0])},
        }
        state, key = _select_pretrained_encoder_state(checkpoint, "target")
        self.assertEqual(key, "ppg_encoder")
        self.assertEqual(state["weight"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
