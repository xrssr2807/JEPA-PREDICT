import importlib.util
import io
import os
import unittest
from unittest import mock

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "scripts",
    "summarize_p2_core_ablations.py",
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_p2_core_ablations", SCRIPT_PATH
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class NonClosingStringIO(io.StringIO):
    def close(self):
        pass


class P2AblationSummaryTests(unittest.TestCase):
    def _checkpoint(self, test_status="sealed"):
        return {
            "test_status": test_status,
            "experiment_id": "P2_phase2_seed42",
            "encoder_init": "pretrained",
            "patient_mil": True,
            "multiscale": True,
            "multidisease_channel": "both",
            "pretrained_checkpoint": "/weights/jepa_best.pt",
            "data_split": {"sha256": "split-hash"},
            "validation_metrics": {"auc": 0.78, "f1": 0.34},
            "val_chd_auc": 0.81,
        }

    def test_sealed_checkpoint_produces_paper_metrics(self):
        with mock.patch.object(
            SUMMARY.torch, "load", return_value=self._checkpoint()
        ):
            row = SUMMARY.checkpoint_row("model.pt", "phase2", 42)

        self.assertEqual(row["test_status"], "sealed")
        self.assertEqual(row["split_sha256"], "split-hash")
        self.assertAlmostEqual(row["val_macro_auc"], 0.78)
        self.assertAlmostEqual(row["val_chd_auc"], 0.81)
        self.assertAlmostEqual(row["val_f1"], 0.34)

    def test_unsealed_checkpoint_is_rejected(self):
        with mock.patch.object(
            SUMMARY.torch,
            "load",
            return_value=self._checkpoint(test_status="evaluated"),
        ):
            with self.assertRaisesRegex(ValueError, "not test-sealed"):
                SUMMARY.checkpoint_row("model.pt", "phase2", 42)

    def test_paper_summary_explains_delta_direction(self):
        aggregate_rows = [{
            "experiment": "phase2",
            "val_macro_auc_n": 1,
            "val_macro_auc_mean": 0.78,
            "val_macro_auc_std": 0.0,
            "val_chd_auc_mean": 0.81,
            "val_chd_auc_std": 0.0,
            "val_f1_mean": 0.34,
            "val_f1_std": 0.0,
        }]
        delta_rows = [{
            "experiment": "mil_off",
            "seed": 42,
            "delta_val_macro_auc": -0.01,
            "delta_val_chd_auc": -0.02,
            "delta_val_f1": -0.03,
        }]
        buffer = NonClosingStringIO()
        with mock.patch("builtins.open", return_value=buffer):
            SUMMARY.write_paper_summary(
                "summary.md", [{}], aggregate_rows, delta_rows, "split-hash"
            )
        text = buffer.getvalue()

        self.assertIn("测试集保持封存", text)
        self.assertIn("负值表示移除该组件后性能下降", text)
        self.assertIn("mil_off", text)


if __name__ == "__main__":
    unittest.main()
