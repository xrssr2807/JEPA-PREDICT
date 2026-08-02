import unittest

import numpy as np

from dataset.sampling import bridge_device_sampling_rate, polyphase_resample


class SamplingRateBridgeTests(unittest.TestCase):
    def test_duration_is_preserved_across_device_rate(self):
        time = np.arange(1000, dtype=np.float32) / 100.0
        signal = np.sin(2 * np.pi * 1.2 * time)[None, :]
        output, metadata = bridge_device_sampling_rate(
            signal, source_hz=100, device_hz=25, canonical_hz=100
        )
        self.assertEqual(output.shape, (1, 1000))
        self.assertAlmostEqual(metadata["duration_seconds"], 10.0)
        self.assertEqual(metadata["device_hz"], 25.0)

    def test_polyphase_resample_has_exact_requested_length(self):
        signal = np.arange(13, dtype=np.float32)[None, :]
        output = polyphase_resample(
            signal, source_hz=13, target_hz=7, expected_length=7
        )
        self.assertEqual(output.shape[-1], 7)
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
