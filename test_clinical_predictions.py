import unittest

import numpy as np

from scripts.evaluate_clinical_predictions import (
    expected_calibration_error,
    paired_delong,
)


class ClinicalPredictionTests(unittest.TestCase):
    def test_paired_delong_detects_better_ranking(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        strong = np.asarray([0.05, 0.10, 0.20, 0.75, 0.85, 0.95])
        weak = np.asarray([0.10, 0.80, 0.20, 0.70, 0.30, 0.90])
        result = paired_delong(labels, strong, weak)
        self.assertGreater(result["delta_reference_minus_comparison"], 0.0)
        self.assertGreaterEqual(result["delong_p_two_sided"], 0.0)
        self.assertLessEqual(result["delong_p_two_sided"], 1.0)

    def test_expected_calibration_error_is_zero_for_exact_groups(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.float64)
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.0)


if __name__ == "__main__":
    unittest.main()
