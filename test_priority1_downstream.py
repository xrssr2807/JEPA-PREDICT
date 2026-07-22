import unittest

import numpy as np
import torch
import torch.nn as nn

from dataset.data import MultiDiseasePatientMILDataset
from models.classifier import (
    DiseaseConditionedMILHead,
    DualStreamPatientMILClassifier,
    PatientMILClassifier,
)
from train_downstream import train_epoch


class DummyEncoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, signal, return_all=False):
        base = signal.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        pooled = base.repeat(1, self.dim) * self.scale
        tokens = pooled.unsqueeze(1).repeat(1, 4, 1)
        return pooled, tokens


class DummySegmentDataset:
    def __getitem__(self, index):
        signal = torch.arange(32, dtype=torch.float32).reshape(2, 16) + index
        label = torch.tensor([1.0, 0.0])
        return signal, label, "u1"


class PriorityOneDownstreamTests(unittest.TestCase):
    def test_short_patient_bag_is_zero_padded_and_masked(self):
        dataset = MultiDiseasePatientMILDataset.__new__(MultiDiseasePatientMILDataset)
        dataset.uids = ["u1"]
        dataset.uid_to_indices = {"u1": [0, 1]}
        dataset.max_segments = 4
        dataset.train = False
        dataset.segment_dataset = DummySegmentDataset()
        segments, target, uid, mask = dataset[0]

        self.assertEqual(tuple(segments.shape), (4, 2, 16))
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertTrue(torch.equal(segments[2:], torch.zeros_like(segments[2:])))
        self.assertEqual(uid, "u1")
        self.assertEqual(target.tolist(), [1.0, 0.0])

    def test_masked_segments_do_not_change_mil_output(self):
        torch.manual_seed(7)
        head = DiseaseConditionedMILHead(dim=8, num_classes=3, dropout=0.0).eval()
        representation = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        changed = representation.clone()
        changed[~mask] = 1e4

        logits_a, _, attention = head(representation, segment_mask=mask)
        logits_b, _, _ = head(changed, segment_mask=mask)

        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6))
        self.assertTrue(torch.equal(attention[~mask], torch.zeros_like(attention[~mask])))

    def test_disease_conditioned_dual_fusion_respects_mask(self):
        torch.manual_seed(11)
        model = DualStreamPatientMILClassifier(
            DummyEncoder(), DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0, ppg_channel=0, ecg_channel=1,
            disease_conditioned_fusion=True,
        ).eval()
        signals = torch.randn(2, 4, 2, 16)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        changed = signals.clone()
        changed[~mask] = 1e4

        logits_a, embedding = model(
            signals, segment_mask=mask, return_embedding=True,
        )
        logits_b = model(changed, segment_mask=mask)

        self.assertEqual(tuple(logits_a.shape), (2, 3))
        self.assertEqual(tuple(embedding.shape), (2, 8))
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-5))

    def test_ppg_student_can_slice_a_dual_channel_bag(self):
        model = PatientMILClassifier(
            DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0, input_channel=0,
        ).eval()
        signals = torch.randn(2, 4, 2, 16)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits, embedding = model(
            signals, segment_mask=mask, return_embedding=True,
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(embedding.shape), (2, 8))

    def test_legacy_dual_fusion_state_remains_available(self):
        model = DualStreamPatientMILClassifier(
            DummyEncoder(), DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0,
            disease_conditioned_fusion=False,
        )
        keys = model.state_dict().keys()
        self.assertIn("modality_gate.weight", keys)
        self.assertIn("mil_head.weight", keys)
        self.assertFalse(any(key.startswith("modality_mil_head.") for key in keys))

    def test_dual_teacher_distillation_step_is_finite(self):
        student = PatientMILClassifier(
            DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0, input_channel=0,
        )
        teacher = DualStreamPatientMILClassifier(
            DummyEncoder(), DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0,
            disease_conditioned_fusion=True,
        )
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        batch = (
            torch.randn(2, 4, 2, 16),
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            ["u1", "u2"],
            torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        )
        loss, _ = train_epoch(
            student, [batch], optimizer, nn.BCEWithLogitsLoss(),
            torch.device("cpu"), multilabel=True,
            teacher_model=teacher, teacher_logit_weight=0.3,
            teacher_embedding_weight=0.1, teacher_temperature=2.0,
        )
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
