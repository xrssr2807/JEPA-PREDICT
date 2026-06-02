"""
Physiological signal augmentation (PhysioAugment).

Domain-specific augmentations for ECG/PPG signals:
  - Amplitude scaling
  - Gaussian jitter
  - Baseline wander (simulated respiratory drift)
  - Time shift (simulated sensor displacement)

Adapted from CWT-MAE v3 (Wearable-Foundation-Model).
"""
import numpy as np


class PhysioAugment:
    """
    Realistic physiological signal augmentations.

    All augmentations operate on numpy arrays of shape (C, L) where
    C is the number of channels (usually 1) and L is signal length.

    Parameters
    ----------
    jitter_std : float
        Standard deviation of Gaussian noise added to the signal.
    scale_range : tuple
        (min, max) for uniform amplitude scaling.
    max_shift : int
        Maximum number of samples to shift the signal (simulates
        temporal misalignment / sensor displacement).
    wander_amp : float
        Amplitude of the sinusoidal baseline wander (simulates
        respiratory drift and low-frequency motion artifact).
    apply_prob : float
        Probability of applying each augmentation independently.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        jitter_std: float = 0.02,
        scale_range: tuple = (0.85, 1.15),
        max_shift: int = 50,
        wander_amp: float = 0.05,
        apply_prob: float = 0.8,
        seed: int = None,
    ):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.wander_amp = wander_amp
        self.apply_prob = apply_prob
        self.rng = np.random.RandomState(seed)

    def __repr__(self):
        return (
            f"PhysioAugment(jitter={self.jitter_std}, scale={self.scale_range}, "
            f"shift={self.max_shift}, wander={self.wander_amp}, p={self.apply_prob})"
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to input signal.

        Args:
            x: (C, L) numpy array

        Returns:
            augmented: (C, L) numpy array
        """
        # Work on a copy
        x = x.copy().astype(np.float64)
        C, L = x.shape

        # --- 1. Amplitude scaling ---
        if self.rng.rand() < self.apply_prob:
            s = self.rng.uniform(*self.scale_range, size=(C, 1))
            x = x * s

        # --- 2. Gaussian jitter ---
        if self.rng.rand() < self.apply_prob:
            noise = self.rng.randn(*x.shape) * self.jitter_std
            x = x + noise

        # --- 3. Baseline wander (sinusoidal) ---
        if self.rng.rand() < self.apply_prob:
            t = np.linspace(0, 1, L)
            for c in range(C):
                freq = self.rng.uniform(0.5, 2.0)  # low-frequency drift
                phase = self.rng.uniform(0, 2 * np.pi)
                wander = self.wander_amp * np.sin(2 * np.pi * freq * t + phase)
                x[c] = x[c] + wander

        # --- 4. Time shift (roll) ---
        if self.rng.rand() < self.apply_prob:
            shift = self.rng.randint(-self.max_shift, self.max_shift + 1)
            x = np.roll(x, shift, axis=-1)

        return x.astype(np.float32)
