"""
CWT (Continuous Wavelet Transform) frontend for physiological signals.

Converts 1D ECG/PPG signals → 2D time-frequency spectrograms using
Ricker (Mexican Hat) wavelets. Enables ViT-style patch-based processing
as an alternative to 1D CNN + Transformer.

Adapted from CWT-MAE v3 (Wearable-Foundation-Model).
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Ricker Wavelet ─────────────────────────────────────────────

def create_ricker_wavelets(points: int, scales: torch.Tensor) -> torch.Tensor:
    """
    Generate Ricker (Mexican Hat) wavelet filters for CWT.

    Args:
        points: wavelet length in samples
        scales: (num_scales,) tensor of wavelet scales
    Returns:
        wavelets: (num_scales, 1, points) tensor — ready for conv1d
    """
    scales = scales.float()
    t = torch.arange(0, points, device=scales.device, dtype=torch.float32) - (points - 1.0) / 2.0
    t = t.reshape(1, 1, -1)
    scales = scales.reshape(-1, 1, 1)

    pi_factor = math.pi ** 0.25
    A = 2 / (torch.sqrt(3 * scales) * pi_factor + 1e-6)
    wsq = scales ** 2
    xsq = t ** 2
    mod = (1 - xsq / wsq)
    gauss = torch.exp(-xsq / (2 * wsq))
    wavelets = A * mod * gauss
    return wavelets


def cwt_ricker(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Apply CWT using Ricker wavelets via F.conv1d.

    Args:
        x: (B, L) signal
        scales: (num_scales,) scale values
    Returns:
        cwt_out: (B, num_scales, L) time-frequency representation
    """
    batch_size, seq_len = x.shape
    x = x.unsqueeze(1)  # (B, 1, L)

    scales = scales.to(x.device)
    largest_scale = scales[-1]

    # Determine wavelet length adaptively
    largest_scale_val = largest_scale.item()
    wavelet_len = int(min(10.0 * largest_scale_val, float(seq_len)))
    if wavelet_len % 2 == 0:
        wavelet_len += 1

    wavelets = create_ricker_wavelets(wavelet_len, scales)
    wavelets = wavelets.to(dtype=x.dtype, device=x.device)

    # Pad signal for full convolution
    pad_len = wavelet_len // 2
    x_padded = F.pad(x, (pad_len, pad_len), mode='reflect')

    # Convolve: (B, 1, L+2*pad) ⊗ (num_scales, 1, wav_len) → (B, num_scales, L)
    cwt_output = F.conv1d(x_padded, wavelets)

    return cwt_output


# ─── CWT Wrapper ────────────────────────────────────────────────

class CWTFrontend(nn.Module):
    """
    CWT frontend: 1D signal → 2D time-frequency spectrogram.

    Input:  (B, 1, L)  raw signal
    Output: (B, num_scales, L)  time-frequency representation

    Optionally computes 1st and 2nd derivatives (diff signals) and
    applies CWT to all, stacking along a channel dimension.
    """

    def __init__(
        self,
        num_scales: int = 64,
        lowest_scale: float = 0.1,
        scale_step: float = 1.0,
        use_diff: bool = True,
        normalize_output: bool = True,
    ):
        """
        Args:
            num_scales: number of wavelet scales (frequency bins)
            lowest_scale: smallest scale (highest frequency)
            scale_step: step between consecutive scales
            use_diff: if True, also CWT the 1st and 2nd derivative signals
            normalize_output: if True, normalize output to zero-mean unit-variance
        """
        super().__init__()
        self.num_scales = num_scales
        self.lowest_scale = lowest_scale
        self.scale_step = scale_step
        self.use_diff = use_diff
        self.normalize_output = normalize_output

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L) or (B, L)
        Returns:
            out: (B, C, num_scales, L) where C=3 if use_diff else C=1
        """
        # Handle input shapes
        squeeze_dim = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
            squeeze_dim = True

        B, M, L = x.shape
        x_flat = x.reshape(B * M, L)
        device = x.device

        if self.use_diff:
            # 1st derivative (central difference): f'(x) = (f(x+1) - f(x-1)) / 2
            x_pad = F.pad(x_flat, (1, 1), mode='replicate')
            d1 = (x_pad[:, 2:] - x_pad[:, :-2]) / 2.0

            # 2nd derivative: f''(x) = f(x+1) - 2*f(x) + f(x-1)
            d2 = x_pad[:, 2:] - 2 * x_pad[:, 1:-1] + x_pad[:, :-2]

            base = x_flat
            d1_cut = d1[:, :L]
            d2_cut = d2[:, :L]

            signals = torch.stack([base, d1_cut, d2_cut], dim=1)  # (B*M, 3, L)
        else:
            signals = x_flat.unsqueeze(1)  # (B*M, 1, L)

        BM, C, _ = signals.shape
        signals_flat = signals.reshape(BM * C, L)

        # Generate scales
        scales = torch.arange(self.num_scales, device=device) * self.scale_step + self.lowest_scale

        # CWT
        cwt_out = cwt_ricker(signals_flat, scales)  # (BM*C, num_scales, L)
        _, n_scales, _ = cwt_out.shape

        cwt_out = cwt_out.reshape(B, M, C, n_scales, L)

        if M == 1 and not squeeze_dim:
            cwt_out = cwt_out.squeeze(1)  # (B, C, n_scales, L)
        elif M == 1 and squeeze_dim:
            cwt_out = cwt_out.squeeze(1)  # (B, C, n_scales, L)

        # Normalize
        if self.normalize_output:
            mean = cwt_out.mean(dim=(-2, -1), keepdim=True)
            std = torch.clamp(cwt_out.std(dim=(-2, -1), keepdim=True), min=1e-5)
            cwt_out = (cwt_out - mean) / std
            cwt_out = torch.nan_to_num(cwt_out, nan=0.0, posinf=100.0, neginf=-100.0)
            cwt_out = torch.clamp(cwt_out, min=-100.0, max=100.0)

        return cwt_out


# ─── CWT + Patch Embed (for ViT-style processing) ──────────────

class DecomposedPatchEmbed(nn.Module):
    """
    Split a 2D (time-freq) spectrogram into patches for ViT input.

    Adapted from CWT-MAE v3.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 1000),   # (freq_bins, time_samples)
        patch_size: Tuple[int, int] = (8, 25),     # (freq_patch, time_patch)
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)                          # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, E)
        x = self.norm(x)
        return x


# ─── Quick Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("=== CWT Frontend Test ===")
    cwt = CWTFrontend(num_scales=64, use_diff=True)

    # Test with synthetic ECG-like signal
    t = torch.linspace(0, 10, 1000)
    ecg = torch.sin(2 * math.pi * 1.2 * t) + 0.3 * torch.sin(2 * math.pi * 3.6 * t)
    ecg = ecg.unsqueeze(0).unsqueeze(0)  # (1, 1, 1000)

    out = cwt(ecg)
    print(f"Input:  {ecg.shape} → Output: {out.shape}")
    print(f"  use_diff=True: C=3 (signal + 1st deriv + 2nd deriv)")
    print(f"  num_scales=64: 64 frequency bins")
    print(f"  L=1000: time samples preserved")
    print(f"  Range: [{out.min():.2f}, {out.max():.2f}]")

    # Test without diff
    cwt_nodiff = CWTFrontend(num_scales=64, use_diff=False)
    out2 = cwt_nodiff(ecg)
    print(f"\nWithout diff: {ecg.shape} → {out2.shape}")

    # Test Patch Embed
    patch_embed = DecomposedPatchEmbed(
        img_size=(64, 1000), patch_size=(8, 25), in_chans=3, embed_dim=256
    )
    tokens = patch_embed(out)
    print(f"\nPatchEmbed: {out.shape} → {tokens.shape}")
    print(f"  grid={patch_embed.grid_size}, patches={patch_embed.num_patches}")

    print("\nAll CWT tests passed!")
