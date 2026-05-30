"""
JEPA: Joint Embedding Predictive Architecture for ECG→PPG cross-channel prediction.

Pre-training task:
    Given ECG (context), predict the embedding of PPG (target).
    The predictor samples a latent variable z to handle multi-modality.

Context Encoder (ECG) ──→ Embedding ──→ Predictor(s_x, z) ──→ predicted s_y
                                                                     │
Target Encoder (PPG)  ──→ Embedding ───────────────────────▶ L2 loss
  (EMA updated, stop_gradient)
"""
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SignalEncoder


def ema_update(student: nn.Module, teacher: nn.Module, momentum: float):
    """Exponential moving average update of teacher from student."""
    with torch.no_grad():
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)


def cosine_schedule(start: float, end: float, progress: float) -> float:
    """Cosine schedule: starts at `start`, moves to `end`."""
    return end + (start - end) * 0.5 * (1 + torch.cos(
        torch.tensor(progress * 3.141592653589793)
    )).item()


class Predictor(nn.Module):
    """
    Predictor network: takes context embedding + latent variable,
    predicts target embedding.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 128,
        latent_dim: int = 32,
    ):
        super().__init__()

        # Project context embedding down to prediction space
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # The predictor body (takes hidden + latent_z, outputs prediction)
        total_input = hidden_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self.latent_dim = latent_dim

    def forward(
        self, context_embed: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_embed: (B, input_dim) — from context encoder
            z: (B, latent_dim) or None — latent variable (sampled if None)
        Returns:
            pred: (B, output_dim) — predicted target embedding
            z: (B, latent_dim) — the latent variable used
        """
        if z is None:
            z = torch.randn(
                context_embed.size(0), self.latent_dim,
                device=context_embed.device
            )

        # Project context
        h = self.proj(context_embed)  # (B, hidden_dim)

        # Concatenate with latent
        h = torch.cat([h, z], dim=-1)  # (B, hidden_dim + latent_dim)

        # Predict
        pred = self.net(h)  # (B, output_dim)

        return pred, z


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture.

    Two encoders with identical architecture:
        - context_encoder (ECG): receives gradients
        - target_encoder (PPG): updated via EMA, stop_gradient in loss

    During pre-training, only context_encoder and predictor receive gradients.
    """

    def __init__(
        self,
        # Encoder config
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
        # JEPA config
        embedding_dim: int = 128,
        predictor_hidden: int = 128,
        latent_dim: int = 32,
        num_latent_samples: int = 4,
        ema_momentum: float = 0.996,
    ):
        super().__init__()

        # Two encoders with identical architecture
        encoder_kwargs = dict(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            transformer_layers=transformer_layers,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            transformer_dropout=transformer_dropout,
            max_seq_len=max_seq_len,
            pool_type=pool_type,
        )

        self.context_encoder = SignalEncoder(**encoder_kwargs)
        self.target_encoder = SignalEncoder(**encoder_kwargs)

        # Copy weights: target starts identical to context
        self.target_encoder.load_state_dict(
            copy.deepcopy(self.context_encoder.state_dict())
        )

        # Stop gradients through target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Project context embedding to match target embedding
        self.context_proj = nn.Sequential(
            nn.Linear(transformer_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(transformer_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        # Copy projection weights too
        self.target_proj.load_state_dict(
            copy.deepcopy(self.context_proj.state_dict())
        )
        for param in self.target_proj.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = Predictor(
            input_dim=transformer_dim,
            hidden_dim=predictor_hidden,
            output_dim=embedding_dim,
            latent_dim=latent_dim,
        )

        self.latent_dim = latent_dim
        self.num_latent_samples = num_latent_samples
        self.ema_momentum = ema_momentum
        self.embedding_dim = embedding_dim
        self.transformer_dim = transformer_dim

    def update_target_encoder(self, momentum: float):
        """EMA update target encoder towards context encoder."""
        ema_update(self.context_encoder, self.target_encoder, momentum)
        ema_update(self.context_proj, self.target_proj, momentum)

    def forward_context(self, x: torch.Tensor) -> torch.Tensor:
        """Encode context signal (ECG). Returns pooled embedding."""
        embed, _ = self.context_encoder(x)
        return embed  # (B, transformer_dim)

    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Encode target signal (PPG). Returns projected embedding (no grad)."""
        with torch.no_grad():
            embed, _ = self.target_encoder(x)
            embed = self.target_proj(embed)
        return embed  # (B, embedding_dim)

    def forward(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass.

        Args:
            ecg: (B, 1, L) — context signal
            ppg: (B, 1, L) — target signal
            z: (B, latent_dim) or None
        Returns:
            pred: (B, embedding_dim) — predicted PPG embedding
            target: (B, embedding_dim) — actual PPG embedding (no grad)
            context_embed: (B, transformer_dim)
        """
        # Context
        context_embed = self.forward_context(ecg)

        # Predict target embedding with latent variable
        pred, z = self.predictor(context_embed, z)

        # Target (no grad)
        target_embed = self.forward_target(ppg)

        return pred, target_embed, context_embed

    def compute_loss(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute JEPA loss with multi-latent sampling.

        For each sample, we draw `num_latent_samples` latent vectors,
        and take the minimum prediction error across samples.

        Args:
            ecg: (B, 1, L)
            ppg: (B, 1, L)
        Returns:
            loss: scalar tensor
            info: dict with metrics
        """
        B = ecg.size(0)

        # Get context embedding once
        context_embed = self.forward_context(ecg)  # (B, transformer_dim)

        # Get target embedding once (deterministic)
        target_embed = self.forward_target(ppg)  # (B, embedding_dim)

        # Sample multiple latent variables, compute loss for each
        all_losses = []
        best_z = None
        best_loss_per_sample = torch.full(
            (B,), float("inf"), device=ecg.device
        )

        for _ in range(self.num_latent_samples):
            z = torch.randn(B, self.latent_dim, device=ecg.device)
            pred, _ = self.predictor(context_embed, z)
            loss_per_sample = F.mse_loss(pred, target_embed, reduction="none").mean(dim=-1)

            # Track best z per sample
            improved = loss_per_sample < best_loss_per_sample
            if best_z is None:
                best_z = z.clone()
            else:
                best_z[improved] = z[improved].clone()
            best_loss_per_sample = torch.minimum(best_loss_per_sample, loss_per_sample)

            all_losses.append(loss_per_sample.mean())

        loss = best_loss_per_sample.mean()

        info = {
            "loss": loss.item(),
            "best_loss": best_loss_per_sample.mean().item(),
            "all_sample_losses": [l.item() for l in all_losses],
        }

        return loss, info
