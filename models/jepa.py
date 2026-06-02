"""
JEPA: Joint Embedding Predictive Architecture for ECG→PPG cross-channel prediction.

Pre-training task:
    Given ECG (context), predict the embedding of PPG (target).
    The predictor samples a latent variable z to handle multi-modality.

Context Encoder (ECG) ──→ Embedding ──→ Predictor(s_x, z) ──→ predicted s_y
                                                                     │
Target Encoder (PPG)  ──→ Embedding ───────────────────────▶ L2 loss
  (EMA updated, stop_gradient)

New additions (from CWT-MAE v3):
  - StatsPredHead: auxiliary task predicting 16 physiological statistics
  - CMAE-style contrastive loss with teacher EMA projector
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


class ProjectionHead(nn.Module):
    """BYOL-style MLP projection head."""
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


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


class StatsPredHead(nn.Module):
    """Predict 16 physiological statistics from the latent representation."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, num_stats: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_stats),
        )
        # Running statistics for target normalization
        self.register_buffer('running_mean', torch.zeros(num_stats))
        self.register_buffer('running_var', torch.ones(num_stats))
        self.momentum = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → (B, num_stats)"""
        return self.net(x)

    def update_stats(self, targets: torch.Tensor):
        """Update running statistics from batch targets."""
        with torch.no_grad():
            batch_mean = targets.mean(dim=0)
            batch_var = targets.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets to zero-mean unit-variance."""
        normed = (targets - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return torch.clamp(normed, min=-10.0, max=10.0)


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture.

    Two encoders with identical architecture:
        - context_encoder (ECG): receives gradients
        - target_encoder (PPG): updated via EMA, stop_gradient in loss

    During pre-training, only context_encoder and predictor receive gradients.

    New (from CWT-MAE v3):
        - StatsPredHead: auxiliary task predicting physiological statistics
        - CMAE contrastive loss: BYOL-style with teacher EMA projector
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
        # New: auxiliary loss config
        use_stats_loss: bool = False,
        stats_loss_weight: float = 0.1,
        use_contrast_loss: bool = False,
        contrast_loss_weight: float = 0.1,
        contrast_decay: float = 0.999,
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

        # ── New: Statistics Prediction Head ──
        self.use_stats_loss = use_stats_loss
        self.stats_loss_weight = stats_loss_weight
        if use_stats_loss:
            self.stats_pred_head = StatsPredHead(
                in_dim=transformer_dim, hidden_dim=transformer_dim, num_stats=16
            )

        # ── New: CMAE Contrastive Learning Components ──
        self.use_contrast_loss = use_contrast_loss
        self.contrast_loss_weight = contrast_loss_weight
        self.contrast_decay = contrast_decay
        if use_contrast_loss:
            # Student: context encoder → projector → predictor
            self.student_projector = ProjectionHead(
                in_dim=transformer_dim, hidden_dim=transformer_dim, out_dim=256
            )
            self.student_predictor = ProjectionHead(
                in_dim=256, hidden_dim=256, out_dim=256
            )
            # Teacher: target encoder → projector (EMA updated)
            self.teacher_projector = ProjectionHead(
                in_dim=transformer_dim, hidden_dim=transformer_dim, out_dim=256
            )
            # Copy and freeze teacher projector
            self.teacher_projector.load_state_dict(
                copy.deepcopy(self.student_projector.state_dict())
            )
            for param in self.teacher_projector.parameters():
                param.requires_grad = False

        self.latent_dim = latent_dim
        self.num_latent_samples = num_latent_samples
        self.ema_momentum = ema_momentum
        self.embedding_dim = embedding_dim
        self.transformer_dim = transformer_dim

    def update_target_encoder(self, momentum: float):
        """EMA update target encoder towards context encoder."""
        ema_update(self.context_encoder, self.target_encoder, momentum)
        ema_update(self.context_proj, self.target_proj, momentum)
        if self.use_contrast_loss:
            ema_update(self.student_projector, self.teacher_projector,
                       self.contrast_decay)

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

    def _compute_jepa_loss(self, ecg, ppg, context_embed, target_embed):
        """Core JEPA loss with multi-latent sampling."""
        B = ecg.size(0)
        all_losses = []
        best_z = None
        best_loss_per_sample = torch.full((B,), float("inf"), device=ecg.device)

        for _ in range(self.num_latent_samples):
            z = torch.randn(B, self.latent_dim, device=ecg.device)
            pred, _ = self.predictor(context_embed, z)
            loss_per_sample = F.mse_loss(pred, target_embed, reduction="none").mean(dim=-1)

            improved = loss_per_sample < best_loss_per_sample
            if best_z is None:
                best_z = z.clone()
            else:
                best_z[improved] = z[improved].clone()
            best_loss_per_sample = torch.minimum(best_loss_per_sample, loss_per_sample)
            all_losses.append(loss_per_sample.mean())

        loss = best_loss_per_sample.mean()
        return loss, {"jepa": loss.item(), "all_samples": [l.item() for l in all_losses]}

    def _compute_stats_loss(self, context_embed, stats_target):
        """Auxiliary statistics prediction loss."""
        if stats_target is None:
            return torch.tensor(0.0, device=context_embed.device), {"stats": 0.0}

        context_pooled = context_embed.mean(dim=0, keepdim=True) if context_embed.dim() == 2 else context_embed
        if context_pooled.dim() == 1:
            context_pooled = context_pooled.unsqueeze(0)

        # Pool to (B, D) if needed
        if context_embed.dim() == 2:
            pooled = context_embed
        else:
            pooled = context_embed.mean(dim=1)

        pred_stats = self.stats_pred_head(pooled)

        if self.training:
            self.stats_pred_head.update_stats(stats_target)
        stats_target_norm = self.stats_pred_head.normalize_targets(stats_target)

        loss = F.smooth_l1_loss(pred_stats, stats_target_norm)
        return loss, {"stats": loss.item()}

    def _compute_contrast_loss(self, context_embed, ecg):
        """BYOL-style contrastive loss using teacher EMA."""
        if not hasattr(self, 'student_projector'):
            return torch.tensor(0.0, device=context_embed.device), {"contrast": 0.0}

        # Pool to (B, D)
        if context_embed.dim() == 3:
            pooled = context_embed.mean(dim=1)
        else:
            pooled = context_embed

        # Student path
        z_student = self.student_predictor(self.student_projector(pooled))

        # Teacher path (no grad — encoder is the target_encoder, same input)
        with torch.no_grad():
            t_embed, _ = self.target_encoder(ecg)
            if t_embed.dim() == 3:
                t_pooled = t_embed.mean(dim=1)
            else:
                t_pooled = t_embed
            z_teacher = self.teacher_projector(t_pooled)

        # Cosine similarity loss (2 - 2*cos = 2*(1-cos) in [0, 4])
        loss = 2 - 2 * F.cosine_similarity(
            F.normalize(z_student, dim=-1),
            F.normalize(z_teacher, dim=-1),
            dim=-1,
        ).mean()
        return loss, {"contrast": loss.item()}

    def compute_loss(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        ecg_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total pre-training loss.

        L_total = L_jepa + w_stats * L_stats + w_contrast * L_contrast

        Args:
            ecg: (B, 1, L)
            ppg: (B, 1, L)
            ecg_stats: (B, 16) or None — precomputed physiological statistics
        Returns:
            loss: scalar tensor
            info: dict with all sub-losses
        """
        B = ecg.size(0)

        # Get context embedding once
        context_embed = self.forward_context(ecg)  # (B, transformer_dim)

        # Get target embedding once (deterministic)
        target_embed = self.forward_target(ppg)  # (B, embedding_dim)

        # 1. JEPA prediction loss
        jepa_loss, jepa_info = self._compute_jepa_loss(
            ecg, ppg, context_embed, target_embed
        )
        total_loss = jepa_loss
        info = dict(jepa_info)

        # 2. Statistics prediction loss (auxiliary)
        if self.use_stats_loss and ecg_stats is not None:
            stats_loss, stats_info = self._compute_stats_loss(
                context_embed, ecg_stats
            )
            total_loss = total_loss + self.stats_loss_weight * stats_loss
            info.update(stats_info)

        # 3. Contrastive loss (BYOL-style)
        if self.use_contrast_loss:
            contrast_loss, contrast_info = self._compute_contrast_loss(
                context_embed, ecg
            )
            total_loss = total_loss + self.contrast_loss_weight * contrast_loss
            info.update(contrast_info)

        info["total_loss"] = total_loss.item()
        return total_loss, info
