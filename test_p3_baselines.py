import csv
import io
import unittest
from unittest import mock

import numpy as np
import torch

from models.baselines import ResNet1DEncoder
from train_contrastive_pretrain import contrastive_loss
from train_downstream import save_multilabel_patient_predictions


class P3BaselineTests(unittest.TestCase):
    def test_resnet_encoder_matches_signal_encoder_interface(self):
        encoder = ResNet1DEncoder(
            in_channels=1,
            output_dim=64,
            widths=(16, 32, 64),
            blocks_per_stage=(1, 1, 1),
        )
        pooled, tokens = encoder(
            torch.randn(3, 1, 1000), return_all=True
        )
        self.assertEqual(tuple(pooled.shape), (3, 64))
        self.assertEqual(tokens.shape[0], 3)
        self.assertEqual(tokens.shape[-1], 64)
        self.assertTrue(torch.isfinite(tokens).all())

    def test_contrastive_loss_prefers_correct_pairs(self):
        identity = torch.eye(4)
        paired_loss, paired_accuracy = contrastive_loss(
            identity, identity, temperature=0.1
        )
        shuffled_loss, _ = contrastive_loss(
            identity, identity[[1, 0, 3, 2]], temperature=0.1
        )
        self.assertLess(float(paired_loss), float(shuffled_loss))
        self.assertEqual(float(paired_accuracy), 1.0)

    def test_patient_prediction_export_preserves_uids(self):
        class NonClosingStringIO(io.StringIO):
            def close(self):
                pass

        labels = np.array([[1, 0], [0, 1]], dtype=np.float32)
        probabilities = np.array(
            [[0.8, 0.2], [0.3, 0.9]], dtype=np.float32
        )
        predictions = (probabilities >= 0.5).astype(np.float32)
        output = NonClosingStringIO()
        with mock.patch("builtins.open", return_value=output):
            save_multilabel_patient_predictions(
                "predictions.csv",
                ["patient_a", "patient_b"],
                ["A", "B"],
                labels,
                predictions,
                probabilities,
                split_role="val",
            )
        output.seek(0)
        rows = list(csv.DictReader(output))
        self.assertEqual(
            [row["uid"] for row in rows],
            ["patient_a", "patient_b"],
        )
        self.assertTrue(all(row["split"] == "val" for row in rows))


if __name__ == "__main__":
    unittest.main()
