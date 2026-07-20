"""Regression tests for the fixed multidisease development split."""
import json
import unittest
from unittest import mock

import numpy as np

from generate_multidisease_patient_split import iterative_multilabel_split
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


if __name__ == "__main__":
    unittest.main()
