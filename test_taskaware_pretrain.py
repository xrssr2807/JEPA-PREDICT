"""Focused regression tests for guarded downstream-feedback pre-training."""
import unittest

import torch

from config import Config
from models.classifier import DualStreamPatientMILClassifier
from models.jepa import JEPA
from train_downstream import compute_multidisease_objective
from train_taskaware_pretrain import FocusBalancedBatchSampler, _feedback_step


class TaskAwarePretrainTests(unittest.TestCase):
    @staticmethod
    def _tiny_jepa():
        model = JEPA(
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
            phase2_min_delay_ms=40.0,
            phase2_max_delay_ms=160.0,
            phase2_delay_prior_ms=80.0,
        )
        model.set_phase2_progress(1.0)
        return model

    def test_shared_multidisease_objective_matches_manual_composition(self):
        logits = torch.tensor([
            [0.1, -0.2, 0.3, -0.4, 0.8],
            [-0.3, 0.2, -0.1, 0.4, -0.7],
        ], requires_grad=True)
        targets = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        ])
        criterion = torch.nn.BCEWithLogitsLoss()
        loss, parts = compute_multidisease_objective(
            logits,
            targets,
            criterion,
            focus_label_index=4,
            focus_loss_weight=0.5,
            focus_pos_weight=torch.tensor(2.0),
            focus_auc_loss_weight=0.1,
            return_components=True,
        )
        expected = parts["base"] + 0.5 * parts["focus_bce"] + 0.1 * parts["focus_auc"]
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focus_sampler_puts_both_classes_in_every_batch(self):
        sampler = FocusBalancedBatchSampler(
            [1, 1, 0, 0, 0, 0, 0, 0], batch_size=4, seed=3
        )
        targets = [1, 1, 0, 0, 0, 0, 0, 0]
        for batch in sampler:
            values = {targets[index] for index in batch}
            self.assertEqual(values, {0, 1})

    def test_feedback_step_updates_online_encoder_but_not_teacher(self):
        torch.manual_seed(5)
        config = Config()
        config.train.taskaware_head_warmup_steps = 0
        config.train.taskaware_feedback_encoder_grad_ratio = 0.2
        config.train.taskaware_feedback_grad_clip = 1.0
        config.train.chd_label_index = 4
        model = self._tiny_jepa()
        feedback_model = DualStreamPatientMILClassifier(
            model.context_encoder,
            model.ppg_encoder,
            encoder_dim=16,
            num_classes=9,
            use_multiscale=False,
            dropout=0.0,
            encoder_chunk_size=4,
        )
        head_parameters = list(feedback_model.head_parameters())
        shared_parameters = list(feedback_model.shared_encoder_parameters())
        self.assertFalse({id(p) for p in head_parameters} & {id(p) for p in shared_parameters})

        pretrain_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        head_optimizer = torch.optim.AdamW(head_parameters, lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        signals = torch.randn(4, 2, 2, 64)
        labels = torch.zeros(4, 9)
        labels[:2, 4] = 1.0
        before_online = next(model.context_encoder.parameters()).detach().clone()
        before_teacher = next(model.context_teacher.parameters()).detach().clone()

        values = _feedback_step(
            model,
            feedback_model,
            (signals, labels, ["a", "b", "c", "d"]),
            torch.nn.BCEWithLogitsLoss(),
            torch.tensor(1.0),
            pretrain_optimizer,
            head_optimizer,
            scaler,
            torch.device("cpu"),
            config,
            feedback_step=0,
            pretrain_grad_ema=0.5,
            ema_momentum=1.0,
            use_amp=False,
        )
        after_online = next(model.context_encoder.parameters()).detach()
        after_teacher = next(model.context_teacher.parameters()).detach()
        self.assertFalse(torch.equal(before_online, after_online))
        self.assertTrue(torch.equal(before_teacher, after_teacher))
        self.assertGreater(values["encoder_grad_norm"], 0.0)
        self.assertGreater(values["encoder_grad_scale"], 0.0)
        self.assertLessEqual(values["encoder_grad_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
