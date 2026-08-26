import unittest

import torch

from train_ppg_continued_ssl import (
    corrupt_ppg,
    local_morphology_targets,
    spectral_targets,
)


class PPGContinuedSSLTests(unittest.TestCase):
    def test_corruption_preserves_shape_and_keeps_visible_samples(self):
        torch.manual_seed(4)
        signal = torch.randn(6, 1, 100)
        corrupted, patch_mask = corrupt_ppg(
            signal, mask_ratio=0.8, patch_size=10, noise_std=0.0
        )
        self.assertEqual(corrupted.shape, signal.shape)
        self.assertEqual(tuple(patch_mask.shape), (6, 1, 10))
        self.assertFalse(bool(patch_mask.all(dim=-1).any()))
        self.assertTrue(torch.isfinite(corrupted).all())

    def test_morphology_targets_are_finite_and_scale_invariant(self):
        torch.manual_seed(5)
        signal = torch.randn(5, 1, 256)
        first = local_morphology_targets(signal, 16)
        second = local_morphology_targets(signal * 3.0 + 8.0, 16)
        self.assertEqual(tuple(first.shape), (5, 16, 3))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.allclose(first, second, atol=3e-4, rtol=3e-4))

    def test_spectral_targets_are_normalized_band_powers(self):
        rate = 100.0
        time = torch.arange(1000) / rate
        signal = torch.sin(2.0 * torch.pi * 1.2 * time)[None, None]
        targets = spectral_targets(signal, rate)
        self.assertEqual(tuple(targets.shape), (1, 4))
        self.assertTrue(torch.isfinite(targets).all())
        self.assertGreater(float(targets[0, 0]), 0.95)


if __name__ == "__main__":
    unittest.main()
