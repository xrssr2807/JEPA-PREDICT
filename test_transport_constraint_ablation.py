"""Tests for the Phase 2 Transport constraint-composition ablations."""

import unittest

import torch

from config import Config
from models.jepa import JEPA
from train_pretrain import _phase_checkpoint_metadata


class TransportConstraintAblationTests(unittest.TestCase):
    @staticmethod
    def _model(mode: str) -> JEPA:
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
            phase2_transport_mode=mode,
            phase2_sample_rate_hz=100.0,
            phase2_min_delay_ms=40.0,
            phase2_max_delay_ms=160.0,
            phase2_delay_prior_ms=80.0,
            phase2_delay_head_hidden=12,
            phase2_transport_temperature=0.5,
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "transport_mode"):
            self._model("not_a_mode")

    def test_static_delay_uses_one_segment_policy_on_interior_tokens(self):
        torch.manual_seed(2)
        model = self._model("static_delay")
        state = model._build_phase2_transport(torch.randn(2, 12, 8))
        policy = state["conditional_delay_probabilities"]
        # Sources 0..7 can use every delay bin, so no boundary clipping occurs.
        reference = policy[:, :1].expand(-1, 8, -1)
        self.assertTrue(torch.allclose(policy[:, :8], reference, atol=1e-6))

    def test_fixed_prior_selects_nearest_physiological_bin(self):
        model = self._model("fixed_prior")
        state = model._build_phase2_transport(torch.randn(2, 12, 8))
        # token_ms=40 and prior=80 ms, hence a two-token fixed delay.
        self.assertTrue(torch.allclose(
            state["expected_delay"][:, :10],
            torch.full_like(state["expected_delay"][:, :10], 2.0),
        ))
        self.assertEqual(
            model._phase2_effective_regularizer_weights(),
            {
                "delay_prior": 0.0,
                "monotonic": 0.0,
                "delay_smoothness": 0.0,
                "match_mass": 0.0,
            },
        )

    def test_zero_delay_is_identity_transport(self):
        model = self._model("zero_delay")
        state = model._build_phase2_transport(torch.randn(2, 12, 8))
        identity = torch.eye(12).unsqueeze(0).expand(2, -1, -1)
        self.assertTrue(torch.equal(state["forward_transport"], identity))
        self.assertTrue(torch.equal(state["reverse_transport"], identity))
        self.assertEqual(state["expected_delay"].abs().max().item(), 0.0)

    def test_no_monotonic_only_removes_monotonic_weight(self):
        model = self._model("no_monotonic")
        weights = model._phase2_effective_regularizer_weights()
        self.assertEqual(weights["monotonic"], 0.0)
        self.assertEqual(weights["delay_prior"], model.phase2_delay_prior_weight)
        self.assertEqual(
            weights["delay_smoothness"],
            model.phase2_delay_smoothness_weight,
        )
        self.assertEqual(weights["match_mass"], model.phase2_match_mass_weight)

    def test_token_shuffle_matches_interpretability_control(self):
        torch.manual_seed(7)
        full = self._model("full")
        shuffled = self._model("token_shuffled")
        shuffled.load_state_dict(full.state_dict())
        tokens = torch.randn(2, 12, 8)
        full_policy = full._build_phase2_transport(tokens)[
            "conditional_delay_probabilities"
        ]
        shuffled_state = shuffled._build_phase2_transport(tokens)

        expected = torch.roll(
            full_policy, shifts=max(1, tokens.size(1) // 4), dims=1
        )
        expected = expected * shuffled_state["valid_delay"]
        expected = expected / expected.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        self.assertTrue(torch.allclose(
            shuffled_state["conditional_delay_probabilities"],
            expected,
            atol=1e-6,
        ))

    def test_every_mode_has_finite_training_loss(self):
        for mode in (
            "full",
            "static_delay",
            "fixed_prior",
            "zero_delay",
            "no_monotonic",
            "token_shuffled",
        ):
            with self.subTest(mode=mode):
                model = self._model(mode)
                model.train()
                model.set_phase2_progress(1.0)
                loss, info = model.compute_loss(
                    torch.randn(3, 1, 64),
                    torch.randn(3, 1, 64),
                )
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(info["phase2_transport_mode"], mode)
                loss.backward()

    def test_checkpoint_metadata_records_mode_and_effective_weights(self):
        model = self._model("no_monotonic")
        config = Config()
        config.model.pretrain_phase = 2
        config.model.phase2_transport_mode = "no_monotonic"
        metadata = _phase_checkpoint_metadata(model, config)["phase2_config"]
        self.assertEqual(metadata["transport_mode"], "no_monotonic")
        self.assertEqual(
            metadata["effective_constraint_weights"]["monotonic"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
