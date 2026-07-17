"""Regression tests for the fixed multidisease development split."""
import json
import unittest
from unittest import mock

from train_downstream import load_multidisease_development_split


class MultidiseaseDevelopmentSplitTests(unittest.TestCase):
    def _load(self, manifest, available_files):
        split_path = "/repo/splits/development_split.json"
        with mock.patch(
            "train_downstream.resolve_multidisease_split_file",
            return_value=split_path,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(manifest)),
        ):
            return load_multidisease_development_split(
                split_path, "/data", available_files
            )

    def test_exact_manifest_is_loaded_without_patient_leakage(self):
        train = ["train_patienta_0.pkl", "train_patienta_1.pkl"]
        val = ["train_patientb_0.pkl"]
        ignored = ["train_patientc_0.pkl"]
        loaded_train, loaded_val, resolved = self._load(
            {"train": train, "val": val}, train + val + ignored
        )

        self.assertEqual(loaded_train, sorted(train))
        self.assertEqual(loaded_val, sorted(val))
        self.assertEqual(resolved, "/repo/splits/development_split.json")

    def test_patient_overlap_is_rejected(self):
        train = ["train_patienta_0.pkl"]
        val = ["train_patienta_1.pkl"]
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            self._load({"train": train, "val": val}, train + val)

    def test_missing_manifest_file_is_rejected(self):
        train = ["train_patienta_0.pkl"]
        val = ["train_patientb_0.pkl"]
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            self._load({"train": train, "val": val}, train)


if __name__ == "__main__":
    unittest.main()
