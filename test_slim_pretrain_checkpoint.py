"""Tests for disk-safe encoder-only experiment checkpoints."""

import unittest

import torch

from scripts.slim_pretrain_checkpoint import build_slim_payload
from train_downstream import _select_pretrained_encoder_state


class SlimPretrainCheckpointTests(unittest.TestCase):
    def test_slim_checkpoint_remains_downstream_compatible(self):
        encoder = {"weight": torch.randn(8, 8)}
        checkpoint = build_slim_payload({
            "pretrain_phase": 2,
            "model_state_dict": {"large": torch.randn(128, 128)},
            "optimizer_state_dict": {"state": torch.randn(128, 128)},
            "context_encoder": encoder,
            "ppg_encoder": encoder,
            "target_encoder": encoder,
            "phase2_config": {
                "transport_mode": "physio_v2",
                "reverse_loss_weight": 0.25,
            },
            "seed": 42,
        })

        self.assertNotIn("model_state_dict", checkpoint)
        self.assertNotIn("optimizer_state_dict", checkpoint)
        self.assertEqual(checkpoint["checkpoint_format"], "encoder_eval_slim_v1")
        state, key = _select_pretrained_encoder_state(checkpoint, "target")
        self.assertEqual(key, "ppg_encoder")
        self.assertEqual(state.keys(), encoder.keys())


if __name__ == "__main__":
    unittest.main()
