import io
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from dataset.data import (
    MULTIDISEASE_LABEL_SOURCES,
    MultiDiseasePatientMILDataset,
    multidisease_label_value,
)
from models.classifier import (
    DiseaseConditionedMILHead,
    DualStreamPatientMILClassifier,
    PatientMILClassifier,
    SharedPrivateSegmentAdapter,
)
from models.jepa import SharedPrivateTokenProjector, TokenProjectionHead
from config import Config
from train_downstream import (
    _resolve_downstream_shared_private_config,
    build_downstream_dataloaders,
    compute_multilabel_pos_weight,
    finalize_downstream_model,
    load_pretrained_encoder,
    patient_relation_distillation_loss,
    selective_embedding_distillation_loss,
    selective_multilabel_logit_distillation,
    train_epoch,
    validate_downstream_checkpoint_context,
)


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
    @staticmethod
    def _shared_private_adapter(output_dim=8):
        return SharedPrivateSegmentAdapter(
            token_projector=TokenProjectionHead(8, 8, 6),
            shared_private_projector=SharedPrivateTokenProjector(
                dim=6, private_dim=4, hidden_dim=8,
            ),
            shared_dim=6,
            private_dim=4,
            output_dim=output_dim,
            use_multiscale=False,
            dropout=0.0,
        )

    def test_rare_vascular_labels_are_merged_with_logical_or(self):
        self.assertEqual(
            multidisease_label_value(
                {"下肢动脉硬化闭塞症": 1, "脑卒中（中风）": 0},
                "其他疾病",
            ),
            1.0,
        )
        self.assertEqual(
            multidisease_label_value(
                {"下肢动脉硬化闭塞症": 0, "脑卒中（中风）": 1},
                "其他疾病",
            ),
            1.0,
        )
        self.assertEqual(
            multidisease_label_value({}, "其他疾病"),
            0.0,
        )

    def test_merged_label_is_included_in_positive_weight_statistics(self):
        config = Config()
        merged_label = config.data.multidisease_labels[3]
        source_label = MULTIDISEASE_LABEL_SOURCES[merged_label][0]
        dataset = type("Dataset", (), {
            "files": ["train_patient_0.pkl"],
            "data_dir": "/data",
            "disease_labels": config.data.multidisease_labels,
        })()
        with mock.patch("builtins.open", mock.mock_open()), mock.patch(
            "train_downstream.pickle.load",
            return_value={"label": {source_label: 1}},
        ):
            weights = compute_multilabel_pos_weight(
                dataset, torch.device("cpu")
            )
        self.assertAlmostEqual(float(weights[3]), 0.2, places=6)

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

    def test_dual_stream_mil_ablation_accepts_single_segments(self):
        torch.manual_seed(13)
        model = DualStreamPatientMILClassifier(
            DummyEncoder(), DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0, ppg_channel=0, ecg_channel=1,
            disease_conditioned_fusion=True,
        ).eval()
        signals = torch.randn(5, 2, 16)
        logits, embedding = model(signals, return_embedding=True)

        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertEqual(tuple(embedding.shape), (5, 8))
        self.assertTrue(torch.isfinite(logits).all())

    def test_random_encoder_initialization_does_not_load_checkpoint(self):
        encoder = DummyEncoder()
        with mock.patch(
            "train_downstream.build_encoder", return_value=encoder,
        ), mock.patch("train_downstream.torch.load") as load_mock:
            result = load_pretrained_encoder(
                None,
                Config().model,
                "target",
                torch.device("cpu"),
                initialization="random",
            )

        self.assertIs(result, encoder)
        load_mock.assert_not_called()

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

    def test_shared_private_single_stream_head_uses_private_features(self):
        adapter = self._shared_private_adapter()
        model = PatientMILClassifier(
            DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0,
            shared_private_adapter=adapter,
        )
        signals = torch.randn(2, 3, 1, 16)
        mask = torch.ones(2, 3, dtype=torch.bool)
        logits = model(signals, segment_mask=mask)
        logits.sum().backward()

        private_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in adapter.shared_private_projector.private_projector.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertGreater(private_gradient, 0.0)

    def test_shared_private_dual_stream_head_is_finite(self):
        model = DualStreamPatientMILClassifier(
            DummyEncoder(), DummyEncoder(), encoder_dim=8, num_classes=3,
            use_multiscale=False, dropout=0.0,
            ecg_shared_private_adapter=self._shared_private_adapter(),
            ppg_shared_private_adapter=self._shared_private_adapter(),
        ).eval()
        signals = torch.randn(2, 3, 2, 16)
        mask = torch.ones(2, 3, dtype=torch.bool)
        logits, embedding = model(
            signals, segment_mask=mask, return_embedding=True,
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(embedding.shape), (2, 8))
        self.assertTrue(torch.isfinite(logits).all())

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

    def test_target_agreement_gate_rejects_wrong_teacher_decisions(self):
        student_logits = torch.zeros(2, 2, requires_grad=True)
        teacher_logits = torch.tensor([
            [4.0, -4.0],
            [-4.0, 4.0],
        ])
        targets = torch.tensor([
            [1.0, 1.0],
            [0.0, 0.0],
        ])
        loss, reliability, metrics = selective_multilabel_logit_distillation(
            student_logits,
            teacher_logits,
            targets,
            temperature=2.0,
            gate_mode="target_agreement",
            confidence_threshold=0.6,
            focus_label_index=0,
            focus_weight=2.0,
            balance_targets=True,
        )

        self.assertGreater(float(reliability[:, 0].min()), 0.9)
        self.assertTrue(torch.equal(
            reliability[:, 1], torch.zeros_like(reliability[:, 1])
        ))
        self.assertAlmostEqual(
            float(metrics["selected_fraction"]), 0.5, places=6
        )
        loss.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())

    def test_selective_embedding_loss_ignores_rejected_patients(self):
        student = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        reliability = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        loss = selective_embedding_distillation_loss(
            student, teacher, reliability
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_patient_relation_distillation_matches_teacher_geometry(self):
        teacher = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        ])
        reliability = torch.ones(3, 2)
        identical = teacher.clone().requires_grad_(True)
        identical_loss = patient_relation_distillation_loss(
            identical, teacher, reliability
        )
        changed = torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        ], requires_grad=True)
        changed_loss = patient_relation_distillation_loss(
            changed, teacher, reliability
        )

        self.assertAlmostEqual(float(identical_loss), 0.0, places=6)
        self.assertGreater(float(changed_loss), 0.0)
        changed_loss.backward()
        self.assertTrue(torch.isfinite(changed.grad).all())

    def test_selective_relation_teacher_training_step_is_finite(self):
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
            torch.randn(4, 4, 2, 16),
            torch.tensor([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            ["u1", "u2", "u3", "u4"],
            torch.ones(4, 4, dtype=torch.bool),
        )
        loss, _ = train_epoch(
            student, [batch], optimizer, nn.BCEWithLogitsLoss(),
            torch.device("cpu"), multilabel=True,
            teacher_model=teacher,
            teacher_logit_weight=0.3,
            teacher_embedding_weight=0.1,
            teacher_relation_weight=0.1,
            teacher_temperature=2.0,
            teacher_gate_mode="target_agreement",
            teacher_confidence_threshold=0.5,
            teacher_focus_weight=2.0,
            teacher_balance_targets=True,
        )
        self.assertTrue(np.isfinite(loss))

    def test_sealed_development_never_evaluates_test_loader(self):
        config = Config()
        model = nn.Linear(1, 1)
        labels = np.zeros(
            (2, len(config.data.multidisease_labels)), dtype=np.int64
        )
        labels[0, :] = 1
        probs = np.full(labels.shape, 0.5, dtype=np.float32)
        predictions = np.zeros_like(labels)
        auc_list = [0.5] * labels.shape[1]
        evaluation = (
            0.5, 75.0, 0.5, auc_list, 0.0, 0.0, 0.0, 0.0,
            "validation report", predictions, labels, probs,
        )
        best_state = {
            "model_state_dict": model.state_dict(),
            "val_acc": 75.0,
            "val_auc": 0.5,
            "val_f1": 0.0,
            "val_chd_auc": 0.5,
        }

        config.output_dir = "."
        with mock.patch(
            "train_downstream.evaluate_multilabel",
            return_value=evaluation,
        ) as evaluate_mock, mock.patch(
            "train_downstream.tune_thresholds_from_config",
            return_value=np.full(labels.shape[1], 0.5),
        ), mock.patch(
            "train_downstream.save_torch_checkpoint_atomic",
        ) as save_mock, mock.patch(
            "train_downstream.save_multilabel_patient_predictions",
        ):
            finalize_downstream_model(
                model=model,
                best_state=best_state,
                val_loader="validation_loader",
                test_loader="sealed_test_loader",
                criterion=None,
                device=torch.device("cpu"),
                config=config,
                dataset="multidisease",
                num_classes=labels.shape[1],
                focus_idx=4,
                use_dual=False,
                use_amp=False,
                log_fh=io.StringIO(),
                evaluate_test=False,
            )

        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertEqual(
            evaluate_mock.call_args.args[1], "validation_loader"
        )
        checkpoint = save_mock.call_args.args[0]
        self.assertFalse(checkpoint["test_evaluated"])
        self.assertEqual(checkpoint["test_status"], "sealed")

    def test_sealed_dataloader_never_constructs_test_dataset(self):
        config = Config()
        config.data.multidisease_patient_mil = False
        config.train.downstream_batch_size = 1
        config.train.dataloader_workers = 0

        filenames = ["train_u1_0.pkl", "test_u2_0.pkl", "test_u3_0.pkl"]
        config.data.multidisease_dir = "mock_data"
        config.data.multidisease_split_file = "mock_split.json"

        class SealedDataset:
            def __init__(self, **kwargs):
                self.files = list(kwargs.get("files") or [])
                self.split = kwargs["split"]

            def __len__(self):
                return len(self.files)

            def __getitem__(self, index):
                return torch.zeros(1, 8), torch.zeros(8), "uid"

        split_files = {
            "train": [filenames[0]],
            "val": [filenames[1]],
        }
        with mock.patch(
            "train_downstream.os.listdir", return_value=filenames,
        ), mock.patch(
            "train_downstream.load_multidisease_named_split_manifest",
            return_value=(split_files, "mock_split.json"),
        ) as manifest_mock, mock.patch(
            "train_downstream.MultiDiseaseDataset", side_effect=SealedDataset,
        ) as dataset_mock, mock.patch("sys.stdout", new=io.StringIO()) as stdout:
            train_loader, val_loader, test_loader, train_ds, test_ds = (
                build_downstream_dataloaders(
                    config.data,
                    config.train,
                    dataset="multidisease",
                    seal_test=True,
                )
            )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNone(test_loader)
        self.assertIsNone(test_ds)
        self.assertEqual(len(train_ds), 1)
        self.assertEqual(
            manifest_mock.call_args.args[3], ("train", "val")
        )
        constructed_splits = [
            call.kwargs.get("split") for call in dataset_mock.call_args_list
        ]
        self.assertEqual(constructed_splits, ["train", "val"])
        self.assertIn(
            "strict_validation_only=true test_dataset_constructed=false",
            stdout.getvalue(),
        )

    def test_encoder_only_checkpoint_supports_disabled_shared_private_head(self):
        checkpoint = {
            "phase2_config": {"shared_private_enabled": True},
            "model_state_dict": {
                "context_encoder.scale": torch.ones(()),
                "target_encoder.scale": torch.ones(()),
            },
        }
        self.assertIsNone(
            _resolve_downstream_shared_private_config(checkpoint, "off")
        )
        with mock.patch("sys.stdout", new=io.StringIO()) as stdout:
            self.assertIsNone(
                _resolve_downstream_shared_private_config(checkpoint, "auto")
            )
        self.assertIn("encoder-only", stdout.getvalue())
        with self.assertRaisesRegex(ValueError, "requires projector tensors"):
            _resolve_downstream_shared_private_config(checkpoint, "on")

    def test_final_evaluation_rejects_split_hash_mismatch(self):
        config = Config()
        checkpoint = {
            "disease_labels": list(config.data.multidisease_labels),
            "multidisease_channel": config.data.multidisease_channel,
            "shared_private_head": True,
            "data_split": {"sha256": "old"},
        }
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            validate_downstream_checkpoint_context(
                checkpoint,
                config,
                {"sha256": "new"},
                use_shared_private_head=True,
            )

    def test_final_evaluation_rejects_ablation_mismatch(self):
        config = Config()
        checkpoint = {
            "disease_labels": list(config.data.multidisease_labels),
            "multidisease_channel": config.data.multidisease_channel,
            "shared_private_head": False,
            "encoder_init": "random",
            "patient_mil": False,
            "multiscale": False,
            "data_split": {"sha256": "same"},
        }
        with self.assertRaisesRegex(
            ValueError, "encoder initialization mismatch"
        ):
            validate_downstream_checkpoint_context(
                checkpoint,
                config,
                {"sha256": "same"},
                use_shared_private_head=False,
                encoder_init="pretrained",
            )


if __name__ == "__main__":
    unittest.main()
