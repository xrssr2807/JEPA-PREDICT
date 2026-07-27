"""Regression tests for the fixed multidisease development split."""
import json
import unittest
from unittest import mock

import numpy as np

from generate_multidisease_patient_split import (
    derive_downstream_manifest,
    iterative_multilabel_split,
)
from train_downstream import (
    load_multidisease_split_manifest,
    load_taskaware_multidisease_split_manifest,
)


class MultidiseaseDevelopmentSplitTests(unittest.TestCase):
    def _load(self, manifest, available_files):
        split_path = "/repo/splits/multidisease_patient_split.json"
        with mock.patch(
            "train_downstream.resolve_multidisease_split_file",
            return_value=split_path,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(manifest)),
        ):
            return load_multidisease_split_manifest(
                split_path, "/data", available_files
            )

    def test_exact_manifest_is_loaded_without_patient_leakage(self):
        train = ["train_patienta_0.pkl", "train_patienta_1.pkl"]
        val = ["train_patientb_0.pkl"]
        test = ["test_patientc_0.pkl"]
        ignored = ["train_patientd_0.pkl"]
        loaded_train, loaded_val, loaded_test, resolved = self._load(
            {"train": train, "val": val, "test": test},
            train + val + test + ignored,
        )

        self.assertEqual(loaded_train, sorted(train))
        self.assertEqual(loaded_val, sorted(val))
        self.assertEqual(loaded_test, sorted(test))
        self.assertEqual(resolved, "/repo/splits/multidisease_patient_split.json")

    def test_patient_overlap_is_rejected(self):
        train = ["train_patienta_0.pkl"]
        val = ["train_patientb_0.pkl"]
        test = ["test_patienta_1.pkl"]
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            self._load(
                {"train": train, "val": val, "test": test},
                train + val + test,
            )

    def test_missing_manifest_file_is_rejected(self):
        train = ["train_patienta_0.pkl"]
        val = ["train_patientb_0.pkl"]
        test = ["test_patientc_0.pkl"]
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            self._load(
                {"train": train, "val": val, "test": test}, train + test
            )

    def test_manifest_label_schema_mismatch_is_rejected(self):
        manifest = {
            "metadata": {"disease_labels": ["old_label"]},
            "train": ["train_patienta_0.pkl"],
            "val": ["train_patientb_0.pkl"],
            "test": ["test_patientc_0.pkl"],
        }
        split_path = "/repo/splits/multidisease_patient_split.json"
        with mock.patch(
            "train_downstream.resolve_multidisease_split_file",
            return_value=split_path,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(manifest)),
        ), self.assertRaisesRegex(ValueError, "label schema"):
            load_multidisease_split_manifest(
                split_path,
                "/data",
                manifest["train"] + manifest["val"] + manifest["test"],
                expected_disease_labels=["new_label"],
            )

    def test_iterative_split_has_exact_sizes_and_no_unassigned_patients(self):
        labels = np.asarray([
            [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
            [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        ])
        assignments = iterative_multilabel_split(
            labels, ratios=(0.70, 0.15, 0.15), seed=42
        )
        self.assertEqual(np.bincount(assignments, minlength=3).tolist(), [14, 3, 3])
        self.assertFalse(np.any(assignments < 0))

    def test_iterative_split_is_deterministic(self):
        labels = np.tile(np.eye(3, dtype=np.int64), (10, 1))
        first = iterative_multilabel_split(labels, seed=42)
        second = iterative_multilabel_split(labels, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_taskaware_split_has_four_exact_patient_groups(self):
        labels = np.tile(np.eye(4, dtype=np.int64), (10, 1))
        assignments = iterative_multilabel_split(
            labels, ratios=(0.55, 0.15, 0.15, 0.15), seed=7
        )
        self.assertEqual(
            np.bincount(assignments, minlength=4).tolist(), [22, 6, 6, 6]
        )

    def test_taskaware_manifest_rejects_cross_role_patient_leakage(self):
        manifest = {
            "feedback_train": ["train_patienta_0.pkl"],
            "feedback_meta": ["train_patientb_0.pkl"],
            "val": ["train_patientc_0.pkl"],
            "test": ["test_patienta_1.pkl"],
        }
        files = sum(manifest.values(), [])
        split_path = "/repo/splits/multidisease_taskaware_split.json"
        with mock.patch(
            "train_downstream.resolve_multidisease_split_file",
            return_value=split_path,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(manifest)),
        ), self.assertRaisesRegex(ValueError, "patient leakage"):
            load_taskaware_multidisease_split_manifest(
                split_path, "/data", files
            )

    def test_taskaware_manifest_derives_exact_downstream_roles(self):
        labels = ["label_a", "label_b"]
        taskaware = {
            "metadata": {
                "version": 1,
                "ratios": {
                    "feedback_train": 0.55,
                    "feedback_meta": 0.15,
                    "val": 0.15,
                    "test": 0.15,
                },
                "patient_counts": {
                    "feedback_train": 2,
                    "feedback_meta": 1,
                    "val": 1,
                    "test": 1,
                },
                "file_counts": {
                    "feedback_train": 2,
                    "feedback_meta": 1,
                    "val": 1,
                    "test": 1,
                },
                "disease_labels": labels,
                "positive_patient_counts": {
                    "feedback_train": {"label_a": 1, "label_b": 1},
                    "feedback_meta": {"label_a": 1, "label_b": 0},
                    "val": {"label_a": 0, "label_b": 1},
                    "test": {"label_a": 1, "label_b": 1},
                },
            },
            "feedback_train": [
                "train_patientb_0.pkl",
                "train_patienta_0.pkl",
            ],
            "feedback_meta": ["train_patientc_0.pkl"],
            "val": ["train_patientd_0.pkl"],
            "test": ["test_patiente_0.pkl"],
        }
        downstream = derive_downstream_manifest(taskaware)

        self.assertEqual(
            downstream["train"],
            [
                "train_patienta_0.pkl",
                "train_patientb_0.pkl",
                "train_patientc_0.pkl",
            ],
        )
        self.assertEqual(
            downstream["metadata"]["patient_counts"],
            {"train": 3, "val": 1, "test": 1},
        )
        self.assertEqual(
            downstream["metadata"]["positive_patient_counts"]["train"],
            {"label_a": 2, "label_b": 1},
        )
        self.assertEqual(
            downstream["metadata"]["source_roles"]["train"],
            ["feedback_train", "feedback_meta"],
        )


if __name__ == "__main__":
    unittest.main()
