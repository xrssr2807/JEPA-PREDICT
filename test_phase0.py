"""Regression tests for the Phase 0 trustworthy pre-training baseline."""
import unittest
from unittest import mock

import torch
import torch.nn as nn

from dataset.data import (
    PretrainDatasetPT,
    infer_pretrain_uid,
    split_pretrain_files,
)
from models.jepa import JEPA, StatsPredHead, ema_update
from preprocess import zscore_per_channel


class Phase0DataTests(unittest.TestCase):
    def test_patient_split_is_deterministic_and_disjoint(self):
        files = [
            "combined_processed_data_2d_part100_0.pt",
            "combined_processed_data_2d_part100_1.pt",
            "combined_processed_data_2d_part200_0.pt",
            "combined_processed_data_2d_part200_1.pt",
            "combined_processed_data_2d_part300_0.pt",
        ]
        split_a = split_pretrain_files(files, val_ratio=1 / 3, seed=42)
        split_b = split_pretrain_files(files, val_ratio=1 / 3, seed=42)
        self.assertEqual(split_a, split_b)

        train_uids = {infer_pretrain_uid(name) for name in split_a[0]}
        val_uids = {infer_pretrain_uid(name) for name in split_a[1]}
        self.assertFalse(train_uids & val_uids)
        self.assertEqual(len(split_a[0]) + len(split_a[1]), len(files))

    def test_preprocessed_stats_are_returned_and_required(self):
        valid_sample = {
            "ecg": torch.zeros(1, 32),
            "ppg": torch.ones(1, 32),
            "ecg_stats": torch.arange(16, dtype=torch.float32),
        }
        with mock.patch("dataset.data.os.listdir", return_value=["sample_part1_0.pt"]), \
             mock.patch("dataset.data.torch.load", return_value=valid_sample):
            dataset = PretrainDatasetPT("mock_data", return_stats=True)
            _, _, stats = dataset[0]
            self.assertEqual(tuple(stats.shape), (16,))

        invalid_sample = {"ecg": torch.zeros(1, 32), "ppg": torch.ones(1, 32)}
        with mock.patch("dataset.data.os.listdir", return_value=["sample_part1_0.pt"]), \
             mock.patch("dataset.data.torch.load", return_value=invalid_sample):
            with self.assertRaisesRegex(RuntimeError, "preprocess.py --overwrite"):
                PretrainDatasetPT("mock_data", return_stats=True)

    def test_preprocess_rejects_nonfinite_signal(self):
        signal = torch.zeros(2, 32).numpy()
        signal[0, 3] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            zscore_per_channel(signal)


class Phase0TeacherTests(unittest.TestCase):
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
            transformer_dropout=0.1,
            max_seq_len=32,
            embedding_dim=8,
            predictor_hidden=8,
            latent_dim=4,
            num_latent_samples=2,
        )

    def test_teacher_stays_in_eval_mode(self):
        model = self._tiny_model()
        model.train()
        self.assertTrue(model.context_encoder.training)
        self.assertFalse(model.target_encoder.training)
        self.assertFalse(model.target_proj.training)

    def test_ema_updates_float_and_integer_buffers(self):
        student = nn.BatchNorm1d(4)
        teacher = nn.BatchNorm1d(4)
        student.running_mean.fill_(2.0)
        student.running_var.fill_(3.0)
        student.num_batches_tracked.fill_(7)
        teacher.running_mean.zero_()
        teacher.running_var.fill_(1.0)
        teacher.num_batches_tracked.zero_()

        ema_update(student, teacher, momentum=0.5)
        self.assertTrue(torch.allclose(teacher.running_mean, torch.ones(4)))
        self.assertTrue(torch.allclose(teacher.running_var, torch.full((4,), 2.0)))
        self.assertEqual(teacher.num_batches_tracked.item(), 7)

    def test_stats_running_values_initialize_from_first_batch(self):
        head = StatsPredHead(in_dim=8, hidden_dim=8, num_stats=2)
        targets = torch.tensor([[2.0, 4.0], [4.0, 8.0]])
        head.update_stats(targets)
        self.assertTrue(torch.allclose(
            head.running_mean, targets.double().mean(dim=0)
        ))
        self.assertTrue(
            torch.allclose(
                head.running_var, targets.double().var(dim=0, unbiased=False)
            )
        )
        self.assertEqual(head.num_updates.item(), 1)

    def test_stats_normalization_stays_finite_for_large_targets(self):
        head = StatsPredHead(in_dim=8, hidden_dim=8, num_stats=2)
        targets = torch.tensor([
            [3e30, -3e30],
            [-3e30, 3e30],
        ], dtype=torch.float32)
        head.update_stats(targets)
        normalized = head.normalize_targets(targets)
        self.assertTrue(torch.isfinite(head.running_var).all())
        self.assertTrue(torch.isfinite(normalized).all())


if __name__ == "__main__":
    unittest.main()
