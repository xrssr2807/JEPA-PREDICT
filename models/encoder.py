"""
ECG/PPG Encoder: 1D CNN Stem + Transformer + Adaptive Pooling.

Architecture:
    Input:  (B, 1, L)    L = 3000 (pretrain) or 1000 (downstream)
    CNN:    (B, 256, L//16)
    Trans:  (B, L//16, 256)
    Pool:   (B, 256)
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Positional Encoding ───────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal position encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            (B, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─── SE Block (通道注意力) ──────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 自适应通道加权 (SENet CVPR 2018)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x)

# ─── Inception Block (残差多尺度) ──────────────────────────────────
class InceptionResidual(nn.Module):
    """多尺度并行Conv → 求和融合 → 残差叠加. alpha控制增量强度."""
    def __init__(self, channels: int, stride: int, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, stride=stride, padding=1, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.branch_k7 = nn.Sequential(
            nn.Conv1d(channels, channels, 7, stride=stride, padding=3, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.branch_k15 = nn.Sequential(
            nn.Conv1d(channels, channels, 15, stride=stride, padding=7, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        f = self.branch_k3(x) + self.branch_k7(x) + self.branch_k15(x)
        return self.alpha * f

# ─── CNN Stem ──────────────────────────────────────────────────────
class CNNStem(nn.Module):
    """
    1D CNN backbone + 可选 Inception残差 + SE注意力.

    Default config (4 blocks, stride=2 each):
        (B, 1, 3000) → (B, 64, 1500) → (B, 128, 750) → (B, 256, 375) → (B, 256, 188)
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Tuple[int, ...] = (64, 128, 256, 256),
        kernel_sizes: Tuple[int, ...] = (7, 5, 5, 3),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        use_se: bool = True,
        use_inception: bool = True,
    ):
        super().__init__()

        assert len(channels) == len(kernel_sizes) == len(strides), (
            f"Length mismatch: channels={len(channels)}, "
            f"kernels={len(kernel_sizes)}, strides={len(strides)}"
        )

        self.conv_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.inception_blocks = nn.ModuleList()
        in_ch = in_channels

        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            padding = k // 2
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ))
            # 残差Inception: 叠加在Conv输出上
            if use_inception and out_ch >= 128:
                self.inception_blocks.append(InceptionResidual(out_ch, stride=1, alpha=0.2))
            else:
                self.inception_blocks.append(None)
            # SE 通道注意力
            self.se_blocks.append(SEBlock(out_ch) if use_se else nn.Identity())
            in_ch = out_ch

        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, inc, se in zip(self.conv_blocks, self.inception_blocks, self.se_blocks):
            x = conv(x)
            if inc is not None:
                x = x + inc(x)  # ★ 残差: Conv主路径 + Inception增量
            x = se(x)
        return x


# ─── Transformer Encoder ───────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """Single Transformer block with pre-LN and GELU activation."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-LN
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x, need_weights=False)[0]
        x = self.dropout1(x)
        x = x + residual

        # FFN with pre-LN
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = x + residual

        return x


class TransformerStack(nn.Module):
    """Stack of Transformer encoder blocks with Stochastic Depth."""

    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        layerdrop: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.layerdrop = layerdrop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.training and self.layerdrop > 0:
                if torch.rand(1).item() < self.layerdrop:
                    continue  # ★ Stochastic Depth: 随机跳过
            x = block(x)
        return x


# ─── Full Encoder ──────────────────────────────────────────────────
class SignalEncoder(nn.Module):
    """
    Full encoder: CNN Stem → Permute → Position Encoding → Transformer → Pool.

    Input:  (B, 1, L)
    Output: (B, d_model)  — global representation vector
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: Tuple[int, ...] = (64, 128, 256, 256),
        cnn_kernel_sizes: Tuple[int, ...] = (7, 5, 5, 3),
        cnn_strides: Tuple[int, ...] = (2, 2, 2, 2),
        transformer_layers: int = 4,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        max_seq_len: int = 200,
        pool_type: str = "adaptive_avg",
        layerdrop: float = 0.0,
        use_se: bool = False,
        use_inception: bool = False,
    ):
        super().__init__()

        self.cnn = CNNStem(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            use_se=use_se,
            use_inception=use_inception,
        )

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=transformer_dim,
            max_len=max_seq_len,
            dropout=transformer_dropout,
        )

        # Project CNN output to transformer dim if needed
        cnn_out_ch = cnn_channels[-1]
        self.proj = (
            nn.Linear(cnn_out_ch, transformer_dim)
            if cnn_out_ch != transformer_dim
            else nn.Identity()
        )

        self.transformer = TransformerStack(
            num_layers=transformer_layers,
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            layerdrop=layerdrop,
        )

        self.ln_final = nn.LayerNorm(transformer_dim)

        self.pool_type = pool_type
        self.transformer_dim = transformer_dim

    def forward(
        self, x: torch.Tensor, return_all: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 1, L)
            return_all: if True, also return per-token representations
        Returns:
            pooled: (B, transformer_dim)
            tokens: (B, seq_len, transformer_dim) or None
        """
        # CNN
        x = self.cnn(x)  # (B, C_out, L_out)

        # Permute: (B, C, L) → (B, L, C)
        x = x.transpose(1, 2)  # (B, L_out, C_out)

        # Project to transformer dim
        x = self.proj(x)  # (B, L_out, transformer_dim)

        # Position encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Final LN
        x = self.ln_final(x)

        tokens = x if return_all else None

        # Pool — length-invariant
        if self.pool_type == "adaptive_avg":
            x = x.mean(dim=1)  # (B, transformer_dim)
        elif self.pool_type == "max":
            x = x.max(dim=1)[0]
        elif self.pool_type == "cls":
            # Using first token as CLS (common in Transformers)
            x = x[:, 0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return x, tokens
