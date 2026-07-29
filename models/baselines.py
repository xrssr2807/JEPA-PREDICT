"""Paper baselines with the same encoder interface as SignalEncoder."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Basic residual block used by the supervised ResNet-18 baseline."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride,
            padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class ResNet1DEncoder(nn.Module):
    """ResNet-18-style waveform encoder for a capacity-controlled baseline.

    ``forward(..., return_all=True)`` returns temporal tokens so the baseline
    can reuse exactly the same patient-MIL and multi-scale heads as JEPA.
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 512,
        widths: Tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_stage: Tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        if len(widths) != len(blocks_per_stage):
            raise ValueError("widths and blocks_per_stage must have equal length")
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, widths[0], kernel_size=15, stride=2,
                padding=7, bias=False,
            ),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages = []
        current = widths[0]
        for stage_index, (width, block_count) in enumerate(
            zip(widths, blocks_per_stage)
        ):
            stride = 1 if stage_index == 0 else 2
            blocks = [ResidualBlock1D(current, width, stride=stride)]
            blocks.extend(
                ResidualBlock1D(width, width) for _ in range(block_count - 1)
            )
            stages.append(nn.Sequential(*blocks))
            current = width
        self.stages = nn.Sequential(*stages)
        self.projection = (
            nn.Identity()
            if current == output_dim
            else nn.Conv1d(current, output_dim, kernel_size=1, bias=False)
        )
        self.norm = nn.LayerNorm(output_dim)
        self.transformer_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        token_mask: Optional[torch.Tensor] = None,
        mask_token: Optional[torch.Tensor] = None,
    ):
        if token_mask is not None or mask_token is not None:
            raise ValueError("ResNet1DEncoder does not support token masking")
        x = self.projection(self.stages(self.stem(x)))
        tokens = self.norm(x.transpose(1, 2))
        pooled = tokens.mean(dim=1)
        return pooled, tokens if return_all else None
