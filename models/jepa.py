"""
JEPA: Joint Embedding Predictive Architecture for ECG→PPG cross-channel prediction.

Pre-training task:
    Given ECG (context), predict the embedding of PPG (target).
    The predictor samples a latent variable z to handle multi-modality.

Context Encoder (ECG) ──→ Embedding ──→ Predictor(s_x, z) ──→ predicted s_y
                                                                     │
Target Encoder (PPG)  ──→ Embedding ───────────────────────▶ L2 loss
  (EMA updated, stop_gradient)

Auxiliary objective:
  - StatsPredHead: predicts 16 physiological statistics saved by preprocessing
"""
import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SignalEncoder


def ema_update(student: nn.Module, teacher: nn.Module, momentum: float):
    """EMA-update matching parameters and buffers from student to teacher."""
    with torch.no_grad():
        student_params = dict(student.named_parameters())
        teacher_params = dict(teacher.named_parameters())
        if student_params.keys() != teacher_params.keys():
            raise ValueError("Student and teacher parameter structures do not match")
        for name, param_s in student_params.items():
            param_t = teacher_params[name]
            param_t.mul_(momentum).add_(param_s, alpha=1 - momentum)

        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(teacher.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            raise ValueError("Student and teacher buffer structures do not match")
        for name, buffer_s in student_buffers.items():
            buffer_t = teacher_buffers[name]
            if torch.is_floating_point(buffer_t):
                buffer_t.mul_(momentum).add_(buffer_s, alpha=1 - momentum)
            else:
                buffer_t.copy_(buffer_s)


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


class TokenProjectionHead(nn.Module):
    """LayerNorm projector that operates independently on every token."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenPredictor(nn.Module):
    """Predict cross-modal teacher latents at masked token positions."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedPrivateTokenProjector(nn.Module):
    """Split a modality token into transport-shared and modality-private views.

    The shared branch is a zero-initialized residual adapter. A Phase 2 model
    can therefore initialize this module without immediately changing the
    latent seen by the existing predictors and causal delay head.
    """

    def __init__(self, dim: int, private_dim: int, hidden_dim: int):
        super().__init__()
        self.shared_adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.shared_adapter[-1].weight)
        nn.init.zeros_(self.shared_adapter[-1].bias)
        self.private_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, private_dim),
            nn.LayerNorm(private_dim),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = tokens + self.shared_adapter(tokens)
        private = self.private_projector(tokens)
        return shared, private


class CausalDelayHead(nn.Module):
    """Predict positive ECG-to-PPG delay bins plus unmatched mass per token."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_delay_bins: int,
        unmatched_bias: float = -2.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, num_delay_bins + 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[-1] = float(unmatched_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(self.norm(tokens)))


class PhysiologicalTransportHead(nn.Module):
    """Cross-conditioned ECG/PPG scores for physiological Transport v2.

    The global branch estimates a segment-level delay distribution from both
    modalities. The local branch refines it per ECG token, while the content
    branch scores the actual ECG/PPG token pair at each admissible delay.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        transport_dim: int,
        num_delay_bins: int,
        unmatched_bias: float,
    ):
        super().__init__()
        self.ecg_query = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, transport_dim),
        )
        self.ppg_key = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, transport_dim),
        )
        self.global_delay = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_delay_bins),
        )
        self.local_delay = CausalDelayHead(
            dim,
            hidden_dim,
            num_delay_bins,
            unmatched_bias=unmatched_bias,
        )

        # Start from a physiological prior/local policy, then let paired
        # content become discriminative without destabilizing early training.
        nn.init.normal_(self.ecg_query[-1].weight, std=0.02)
        nn.init.zeros_(self.ecg_query[-1].bias)
        nn.init.normal_(self.ppg_key[-1].weight, std=0.02)
        nn.init.zeros_(self.ppg_key[-1].bias)
        nn.init.zeros_(self.global_delay[-1].weight)
        nn.init.zeros_(self.global_delay[-1].bias)

    def forward(
        self,
        ecg_tokens: torch.Tensor,
        ppg_tokens: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> dict:
        if ecg_tokens.shape != ppg_tokens.shape:
            raise ValueError(
                "Physiological Transport requires shape-matched ECG/PPG tokens"
            )
        batch_size, num_tokens, _ = ecg_tokens.shape
        num_bins = target_indices.size(-1)

        ecg_float = ecg_tokens.float()
        ppg_float = ppg_tokens.float()
        query = F.normalize(self.ecg_query(ecg_float), dim=-1)
        key = F.normalize(self.ppg_key(ppg_float), dim=-1)
        safe_indices = target_indices.clamp(0, num_tokens - 1).expand(
            batch_size, -1, -1
        )
        gathered_key = key.gather(
            1,
            safe_indices.reshape(batch_size, -1).unsqueeze(-1).expand(
                -1, -1, key.size(-1)
            ),
        ).reshape(batch_size, num_tokens, num_bins, key.size(-1))
        content_scores = (
            query.unsqueeze(2) * gathered_key
        ).sum(dim=-1)

        ecg_global = ecg_float.mean(dim=1)
        ppg_global = ppg_float.mean(dim=1)
        global_input = torch.cat(
            (ecg_global, ppg_global, (ecg_global - ppg_global).abs()),
            dim=-1,
        )
        global_logits = self.global_delay(global_input).unsqueeze(1)
        local_logits = self.local_delay(ecg_float)
        return {
            "content_scores": content_scores,
            "global_delay_logits": global_logits,
            "local_delay_logits": local_logits[..., :-1],
            "unmatched_logits": local_logits[..., -1],
        }


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
        # Float64 avoids overflow when raw-device scales vary by many orders.
        self.register_buffer('running_mean', torch.zeros(num_stats, dtype=torch.float64))
        self.register_buffer('running_var', torch.ones(num_stats, dtype=torch.float64))
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long))
        self.momentum = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → (B, num_stats)"""
        return self.net(x)

    def update_stats(self, targets: torch.Tensor):
        """Update running statistics from batch targets."""
        with torch.no_grad():
            if not torch.isfinite(targets).all():
                raise FloatingPointError("Stats targets contain NaN or Inf")
            targets64 = targets.detach().to(dtype=torch.float64)
            batch_mean = targets64.mean(dim=0)
            batch_var = targets64.var(dim=0, unbiased=False)
            if not torch.isfinite(batch_mean).all() or not torch.isfinite(batch_var).all():
                raise FloatingPointError("Stats running moments overflowed")
            if self.num_updates.item() == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.lerp_(batch_mean, self.momentum)
                self.running_var.lerp_(batch_var, self.momentum)
            self.num_updates.add_(1)

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets to zero-mean unit-variance."""
        if not torch.isfinite(targets).all():
            raise FloatingPointError("Stats targets contain NaN or Inf")
        normed = (
            targets.to(dtype=torch.float64) - self.running_mean
        ) / torch.sqrt(self.running_var.clamp_min(0.0) + 1e-5)
        normed = torch.clamp(normed, min=-10.0, max=10.0)
        if not torch.isfinite(normed).all():
            raise FloatingPointError("Normalized stats targets contain NaN or Inf")
        return normed.to(dtype=targets.dtype)


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture.

    Two encoders with identical architecture:
        - context_encoder (ECG): receives gradients
        - target_encoder (PPG): updated via EMA, stop_gradient in loss

    During pre-training, only context_encoder and predictor receive gradients.

    Auxiliary statistics are optional and must be present in the dataset.
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
        # ★ JETS 式掩码：预训练时随机丢弃信号片段
        mask_ratio: float = 0.0,         # 0=关闭, 0.7=保留30%
        mask_patch_size: int = 50,        # 每个patch的采样点数
        # New: auxiliary loss config
        use_stats_loss: bool = False,
        stats_loss_weight: float = 0.1,
        use_token_align: bool = False,
        token_align_weight: float = 0.5,
        token_align_window: int = 3,
        pretrain_phase: int = 0,
        phase1_mask_ratio: float = 0.6,
        phase1_mask_block_tokens: int = 8,
        phase1_bidirectional: bool = True,
        phase1_token_loss_weight: float = 1.0,
        phase2_transport_enabled: bool = True,
        phase2_transport_mode: str = "full",
        phase2_sample_rate_hz: float = 100.0,
        phase2_min_delay_ms: float = 80.0,
        phase2_max_delay_ms: float = 800.0,
        phase2_delay_prior_ms: float = 250.0,
        phase2_delay_head_hidden: int = 128,
        phase2_transport_temperature: float = 0.2,
        phase2_unmatched_bias: float = -2.0,
        phase2_transport_loss_weight: float = 1.0,
        phase2_delay_prior_weight: float = 0.02,
        phase2_monotonic_weight: float = 0.05,
        phase2_delay_smoothness_weight: float = 0.01,
        phase2_match_mass_weight: float = 0.01,
        phase2_target_match_mass: float = 0.95,
        phase2_variance_weight: float = 0.10,
        phase2_covariance_weight: float = 0.01,
        phase2_target_std: float = 0.10,
        phase2_v2_transport_dim: int = 128,
        phase2_v2_content_weight: float = 1.0,
        phase2_v2_global_delay_weight: float = 1.0,
        phase2_v2_local_delay_weight: float = 1.0,
        phase2_v2_sinkhorn_epsilon: float = 1.0,
        phase2_v2_sinkhorn_mass_reg: float = 1.0,
        phase2_v2_sinkhorn_iters: int = 20,
        phase2_counterfactual_weight: float = 0.10,
        phase2_counterfactual_margin: float = 0.10,
        phase2_pat_weak_weight: float = 0.0,
        phase2_shared_private_enabled: bool = False,
        phase2_private_dim: int = 128,
        phase2_shared_private_hidden: int = 256,
        phase2_private_loss_weight: float = 0.50,
        phase2_orthogonality_weight: float = 0.05,
        use_se: bool = False,
        use_inception: bool = False,
    ):
        super().__init__()

        if pretrain_phase not in (0, 1, 2):
            raise ValueError(
                f"Unsupported pretrain_phase={pretrain_phase}; expected 0, 1, or 2"
            )
        self.pretrain_phase = int(pretrain_phase)

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
            use_se=use_se,
            use_inception=use_inception,
        )

        self.context_encoder = SignalEncoder(**encoder_kwargs)

        if self.pretrain_phase == 0:
            self.target_encoder = copy.deepcopy(self.context_encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False

            self.context_proj = nn.Sequential(
                nn.Linear(transformer_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
            self.target_proj = copy.deepcopy(self.context_proj)
            for param in self.target_proj.parameters():
                param.requires_grad = False

            self.predictor = Predictor(
                input_dim=transformer_dim,
                hidden_dim=predictor_hidden,
                output_dim=embedding_dim,
                latent_dim=latent_dim,
            )
        else:
            # Both modalities have an online encoder with direct gradients.
            self.ppg_encoder = copy.deepcopy(self.context_encoder)

            # Each EMA teacher follows the online encoder of the same modality.
            self.context_teacher = copy.deepcopy(self.context_encoder)
            self.target_encoder = copy.deepcopy(self.ppg_encoder)
            for module in (self.context_teacher, self.target_encoder):
                for param in module.parameters():
                    param.requires_grad = False

            self.ecg_token_proj = TokenProjectionHead(
                transformer_dim, transformer_dim, embedding_dim
            )
            self.ppg_token_proj = TokenProjectionHead(
                transformer_dim, transformer_dim, embedding_dim
            )
            self.ecg_teacher_proj = copy.deepcopy(self.ecg_token_proj)
            self.target_proj = copy.deepcopy(self.ppg_token_proj)
            for module in (self.ecg_teacher_proj, self.target_proj):
                for param in module.parameters():
                    param.requires_grad = False

            self.ecg_to_ppg_predictor = TokenPredictor(
                embedding_dim, predictor_hidden
            )
            self.ppg_to_ecg_predictor = TokenPredictor(
                embedding_dim, predictor_hidden
            )
            self.ecg_mask_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
            self.ppg_mask_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
            nn.init.normal_(self.ecg_mask_token, std=0.02)
            nn.init.normal_(self.ppg_mask_token, std=0.02)

            if self.pretrain_phase == 2:
                transport_modes = {
                    "full",
                    "static_delay",
                    "fixed_prior",
                    "zero_delay",
                    "no_monotonic",
                    "token_shuffled",
                    "physio_v2",
                }
                phase2_transport_mode = str(
                    phase2_transport_mode
                ).strip().lower()
                if phase2_transport_mode not in transport_modes:
                    raise ValueError(
                        "Unsupported phase2_transport_mode="
                        f"{phase2_transport_mode!r}; expected one of "
                        f"{sorted(transport_modes)}"
                    )
                if phase2_sample_rate_hz <= 0:
                    raise ValueError("phase2_sample_rate_hz must be positive")
                if phase2_min_delay_ms <= 0:
                    raise ValueError("Phase 2 requires a strictly positive minimum delay")
                if phase2_max_delay_ms < phase2_min_delay_ms:
                    raise ValueError(
                        "phase2_max_delay_ms must be >= phase2_min_delay_ms"
                    )
                if phase2_transport_temperature <= 0:
                    raise ValueError("phase2_transport_temperature must be positive")
                if not 0.0 < phase2_target_match_mass <= 1.0:
                    raise ValueError(
                        "phase2_target_match_mass must be in the interval (0, 1]"
                    )
                if phase2_target_std <= 0:
                    raise ValueError("phase2_target_std must be positive")
                encoder_stride = math.prod(cnn_strides)
                token_ms = 1000.0 * encoder_stride / phase2_sample_rate_hz
                min_delay_tokens = max(
                    1, int(math.ceil(phase2_min_delay_ms / token_ms))
                )
                max_delay_tokens = max(
                    min_delay_tokens,
                    int(math.ceil(phase2_max_delay_ms / token_ms)),
                )
                delay_offsets = torch.arange(
                    min_delay_tokens, max_delay_tokens + 1, dtype=torch.long
                )
                self.register_buffer("phase2_delay_offsets", delay_offsets)
                self.phase2_token_ms = float(token_ms)
                self.phase2_delay_prior_tokens = float(
                    phase2_delay_prior_ms / token_ms
                )
                self.phase2_delay_head = CausalDelayHead(
                    embedding_dim,
                    phase2_delay_head_hidden,
                    delay_offsets.numel(),
                    unmatched_bias=phase2_unmatched_bias,
                )
                self.phase2_transport_temperature = float(
                    phase2_transport_temperature
                )
                self.phase2_transport_enabled = bool(
                    phase2_transport_enabled
                )
                self.phase2_transport_mode = phase2_transport_mode
                self.phase2_transport_loss_weight = float(
                    phase2_transport_loss_weight
                )
                self.phase2_delay_prior_weight = float(phase2_delay_prior_weight)
                self.phase2_monotonic_weight = float(phase2_monotonic_weight)
                self.phase2_delay_smoothness_weight = float(
                    phase2_delay_smoothness_weight
                )
                self.phase2_match_mass_weight = float(phase2_match_mass_weight)
                self.phase2_target_match_mass = float(phase2_target_match_mass)
                self.phase2_variance_weight = float(phase2_variance_weight)
                self.phase2_covariance_weight = float(phase2_covariance_weight)
                self.phase2_target_std = float(phase2_target_std)
                self.phase2_progress = 0.0

                if phase2_v2_transport_dim <= 0:
                    raise ValueError("phase2_v2_transport_dim must be positive")
                if phase2_v2_sinkhorn_epsilon <= 0:
                    raise ValueError(
                        "phase2_v2_sinkhorn_epsilon must be positive"
                    )
                if phase2_v2_sinkhorn_mass_reg <= 0:
                    raise ValueError(
                        "phase2_v2_sinkhorn_mass_reg must be positive"
                    )
                if phase2_v2_sinkhorn_iters <= 0:
                    raise ValueError("phase2_v2_sinkhorn_iters must be positive")
                if phase2_counterfactual_weight < 0:
                    raise ValueError(
                        "phase2_counterfactual_weight must be non-negative"
                    )
                if phase2_counterfactual_margin < 0:
                    raise ValueError(
                        "phase2_counterfactual_margin must be non-negative"
                    )
                if phase2_pat_weak_weight < 0:
                    raise ValueError(
                        "phase2_pat_weak_weight must be non-negative"
                    )
                self.phase2_v2_content_weight = float(
                    phase2_v2_content_weight
                )
                self.phase2_v2_global_delay_weight = float(
                    phase2_v2_global_delay_weight
                )
                self.phase2_v2_local_delay_weight = float(
                    phase2_v2_local_delay_weight
                )
                self.phase2_v2_sinkhorn_epsilon = float(
                    phase2_v2_sinkhorn_epsilon
                )
                self.phase2_v2_sinkhorn_mass_reg = float(
                    phase2_v2_sinkhorn_mass_reg
                )
                self.phase2_v2_sinkhorn_iters = int(
                    phase2_v2_sinkhorn_iters
                )
                self.phase2_counterfactual_weight = float(
                    phase2_counterfactual_weight
                )
                self.phase2_counterfactual_margin = float(
                    phase2_counterfactual_margin
                )
                self.phase2_pat_weak_weight = float(phase2_pat_weak_weight)
                if phase2_transport_mode == "physio_v2":
                    self.phase2_physio_transport = PhysiologicalTransportHead(
                        embedding_dim,
                        phase2_delay_head_hidden,
                        int(phase2_v2_transport_dim),
                        delay_offsets.numel(),
                        unmatched_bias=phase2_unmatched_bias,
                    )

                self.phase2_shared_private_enabled = bool(
                    phase2_shared_private_enabled
                )
                self.phase2_private_loss_weight = float(
                    phase2_private_loss_weight
                )
                self.phase2_orthogonality_weight = float(
                    phase2_orthogonality_weight
                )
                self.phase2_shared_private_progress = (
                    1.0 if self.phase2_shared_private_enabled else 0.0
                )
                if self.phase2_shared_private_enabled:
                    if phase2_private_dim <= 0:
                        raise ValueError("phase2_private_dim must be positive")
                    if phase2_shared_private_hidden <= 0:
                        raise ValueError(
                            "phase2_shared_private_hidden must be positive"
                        )
                    if phase2_private_loss_weight < 0:
                        raise ValueError(
                            "phase2_private_loss_weight must be non-negative"
                        )
                    if phase2_orthogonality_weight < 0:
                        raise ValueError(
                            "phase2_orthogonality_weight must be non-negative"
                        )

                    projector_kwargs = dict(
                        dim=embedding_dim,
                        private_dim=int(phase2_private_dim),
                        hidden_dim=int(phase2_shared_private_hidden),
                    )
                    self.ecg_shared_private = SharedPrivateTokenProjector(
                        **projector_kwargs
                    )
                    self.ppg_shared_private = SharedPrivateTokenProjector(
                        **projector_kwargs
                    )
                    self.ecg_teacher_shared_private = copy.deepcopy(
                        self.ecg_shared_private
                    )
                    self.ppg_teacher_shared_private = copy.deepcopy(
                        self.ppg_shared_private
                    )
                    for module in (
                        self.ecg_teacher_shared_private,
                        self.ppg_teacher_shared_private,
                    ):
                        for param in module.parameters():
                            param.requires_grad = False

                    self.ecg_private_predictor = TokenPredictor(
                        int(phase2_private_dim), predictor_hidden
                    )
                    self.ppg_private_predictor = TokenPredictor(
                        int(phase2_private_dim), predictor_hidden
                    )
        if self.pretrain_phase != 2 and phase2_shared_private_enabled:
            raise ValueError(
                "Shared-private decomposition is currently supported only in Phase 2"
            )

    # ── New: Statistics Prediction Head ──
        self.use_stats_loss = use_stats_loss
        self.stats_loss_weight = stats_loss_weight
        # Phase 1 B2 deliberately excludes delay/transport alignment; that is
        # introduced only in Phase 2 so its contribution remains measurable.
        self.use_token_align = bool(use_token_align and self.pretrain_phase == 0)
        self.token_align_weight = token_align_weight
        self.align_window = int(token_align_window)
        self.phase1_mask_ratio = float(phase1_mask_ratio)
        self.phase1_mask_block_tokens = int(phase1_mask_block_tokens)
        self.phase1_bidirectional = bool(phase1_bidirectional)
        self.phase1_token_loss_weight = float(phase1_token_loss_weight)
        if use_stats_loss:
            self.stats_pred_head = StatsPredHead(
                in_dim=transformer_dim, hidden_dim=transformer_dim, num_stats=16
            )



        self.latent_dim = latent_dim
        self.num_latent_samples = num_latent_samples
        self.ema_momentum = ema_momentum
        self.embedding_dim = embedding_dim
        self.transformer_dim = transformer_dim
        # ★ JETS 掩码参数
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self._enforce_teacher_eval()

    def _enforce_teacher_eval(self):
        """Teacher targets must not depend on dropout or batch statistics."""
        self.target_encoder.eval()
        self.target_proj.eval()
        if self.pretrain_phase >= 1:
            self.context_teacher.eval()
            self.ecg_teacher_proj.eval()
        if (
            self.pretrain_phase == 2
            and getattr(self, "phase2_shared_private_enabled", False)
        ):
            self.ecg_teacher_shared_private.eval()
            self.ppg_teacher_shared_private.eval()

    def train(self, mode: bool = True):
        """Set online modules to ``mode`` while always keeping teachers in eval."""
        super().train(mode)
        self._enforce_teacher_eval()
        return self

    def update_target_encoder(self, momentum: float):
        """Update Phase 0 target or Phase 1 same-modality EMA teachers."""
        if self.pretrain_phase == 0:
            ema_update(self.context_encoder, self.target_encoder, momentum)
            ema_update(self.context_proj, self.target_proj, momentum)
        else:
            ema_update(self.context_encoder, self.context_teacher, momentum)
            ema_update(self.ppg_encoder, self.target_encoder, momentum)
            ema_update(self.ecg_token_proj, self.ecg_teacher_proj, momentum)
            ema_update(self.ppg_token_proj, self.target_proj, momentum)
            if (
                self.pretrain_phase == 2
                and getattr(self, "phase2_shared_private_enabled", False)
            ):
                ema_update(
                    self.ecg_shared_private,
                    self.ecg_teacher_shared_private,
                    momentum,
                )
                ema_update(
                    self.ppg_shared_private,
                    self.ppg_teacher_shared_private,
                    momentum,
                )
        self._enforce_teacher_eval()

    def teacher_student_parameter_cosine(self) -> float:
        """Mean same-modality cosine between online and EMA encoders."""
        pairs = [(self.context_encoder, self.target_encoder)]
        if self.pretrain_phase >= 1:
            pairs = [
                (self.context_encoder, self.context_teacher),
                (self.ppg_encoder, self.target_encoder),
            ]
        cosines = []
        with torch.no_grad():
            for student_module, teacher_module in pairs:
                device = next(student_module.parameters()).device
                dot = torch.zeros((), device=device)
                student_norm = torch.zeros_like(dot)
                teacher_norm = torch.zeros_like(dot)
                teacher_params = dict(teacher_module.named_parameters())
                for name, student in student_module.named_parameters():
                    teacher = teacher_params[name]
                    dot += (student.float() * teacher.float()).sum()
                    student_norm += student.float().square().sum()
                    teacher_norm += teacher.float().square().sum()
                cosines.append(
                    dot / (student_norm.sqrt() * teacher_norm.sqrt()).clamp_min(1e-12)
                )
        return torch.stack(cosines).mean().item()

    def forward_context(self, x: torch.Tensor, return_tokens: bool = False):
        """Encode context signal (ECG).
        Args:
            return_tokens: True 时返回 (pooled, tokens)，减少二次前向
        """
        embed, tokens = self.context_encoder(x, return_all=return_tokens)
        if return_tokens:
            return embed, tokens
        return embed  # (B, transformer_dim)

    def forward_target(self, x: torch.Tensor, return_tokens: bool = False):
        """Encode target signal (PPG). Returns projected embedding (no grad)."""
        with torch.no_grad():
            embed, tokens = self.target_encoder(x, return_all=return_tokens)
            embed = self.target_proj(embed)
        if return_tokens:
            return embed, tokens
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

        if self.pretrain_phase == 0:
            # Predict target embedding with latent variable.
            pred, z = self.predictor(context_embed, z)
        else:
            pred = self.ecg_to_ppg_predictor(
                self.ecg_token_proj(context_embed)
            )

        # Target (no grad)
        target_embed = self.forward_target(ppg)

        return pred, target_embed, context_embed

    def _compute_jepa_loss(self, ecg, ppg, context_embed, target_embed):
        """Core JEPA loss with multi-latent sampling."""
        B = ecg.size(0)
        all_losses = []
        best_z = None
        best_pred = None
        best_loss_per_sample = torch.full((B,), float("inf"), device=ecg.device)

        for _ in range(self.num_latent_samples):
            z = torch.randn(B, self.latent_dim, device=ecg.device)
            pred, _ = self.predictor(context_embed, z)
            loss_per_sample = F.mse_loss(pred, target_embed, reduction="none").mean(dim=-1)

            improved = loss_per_sample < best_loss_per_sample
            if best_z is None:
                best_z = z.clone()
                best_pred = pred.clone()
            else:
                best_z[improved] = z[improved].clone()
                best_pred[improved] = pred[improved].clone()
            best_loss_per_sample = torch.minimum(best_loss_per_sample, loss_per_sample)
            all_losses.append(loss_per_sample.mean())

        loss = best_loss_per_sample.mean()
        pred_std = best_pred.detach().float().std(dim=0, unbiased=False).mean()
        return loss, {
            "jepa": loss.item(),
            "prediction_std": pred_std.item(),
            "all_samples": [l.item() for l in all_losses],
        }

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

    def _compute_token_align_loss(self, ecg=None, ppg=None, token_mask=None,
                                   align_window=3, soft_temperature=0.1,
                                   cached_ctx_tokens=None, cached_tgt_tokens=None):
        """
        ★ Soft-DTW 风格弹性 Token 对齐.

        文档指出: Soft-DTW 允许弹性匹配, 完美吸收 PTT 生理波动。
        ECG_token_i → PPG_window[i-3..i+3] → softmax 加权 (非硬对齐).

        关键优化: 使用 cached_tokens, 避免二次前向导致 OOM。

        Args:
            cached_ctx_tokens: (B, N, D) 缓存的 context encoder token 序列
            cached_tgt_tokens: (B, N, D) 缓存的 target encoder token 序列
            token_mask: (B, 1, N) bool, True=可见位置
            align_window: 单侧搜索窗口 (±3 ≈ ±300ms, 覆盖 PTT)
        """
        if cached_ctx_tokens is not None and cached_tgt_tokens is not None:
            ecg_tokens = cached_ctx_tokens
            ppg_tokens = cached_tgt_tokens.detach()
        else:
            # 兜底: 无缓存时跑前向 (不常用)
            _, ecg_tokens = self.context_encoder(ecg, return_all=True)
            _, ppg_tokens = self.target_encoder(ppg, return_all=True)
            ppg_tokens = ppg_tokens.detach()

        min_n = min(ecg_tokens.size(1), ppg_tokens.size(1))
        if token_mask is not None:
            min_n = min(min_n, token_mask.size(-1))
        ecg_tokens = ecg_tokens[:, :min_n, :]
        ppg_tokens = ppg_tokens[:, :min_n, :]
        N = min_n

        # Soft-DTW 式弹性对齐
        total_loss = 0.0
        valid_count = 0

        for i in range(N):
            # 只在可见 token 上计算

            # 局部搜索窗口: [i-window, i+window]
            start = max(0, i - align_window)
            end = min(N, i + align_window + 1)

            # ECG token_i vs PPG_tokens[start:end] 的余弦相似度
            ecg_tok = ecg_tokens[:, i:i+1, :]  # (B, 1, D)
            ppg_window = ppg_tokens[:, start:end, :]  # (B, W, D)

            sim = F.cosine_similarity(
                ecg_tok.expand(-1, end - start, -1),
                ppg_window, dim=-1
            )  # (B, W)

            # softmax 加权: 自动选择最匹配的 PPG token
            weights = F.softmax(sim / soft_temperature, dim=-1)  # (B, W)

            # 加权后的 PPG 嵌入 (软对齐)
            aligned_ppg = (weights.unsqueeze(-1) * ppg_window).sum(dim=1)  # (B, D)

            # 对齐损失
            align_cos = F.cosine_similarity(ecg_tok.squeeze(1), aligned_ppg, dim=-1)
            loss_i = 1.0 - align_cos
            if token_mask is not None:
                visible = token_mask[:, 0, i].float()
                total_loss += (loss_i * visible).sum()
                valid_count += visible.sum().item()
            else:
                total_loss += loss_i.sum()
                valid_count += ecg_tokens.size(0)

        loss = total_loss / max(valid_count, 1)
        return loss, {"token_align": loss.item(), "window": align_window}

    def freeze_target_encoder(self):
        """冻结 target_encoder (Token 对齐模式: 当做 teacher)."""
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_proj.parameters():
            param.requires_grad = False
        self.target_encoder.eval()
        self.target_proj.eval()
        print("[JEPA] Target encoder frozen (teacher mode)")

    # ═══════════════════════════════════════════════════════════
    # ★ JETS 式掩码：随机丢弃 ~70% 信号patch，强制编码器从局部学习
    # ═══════════════════════════════════════════════════════════

    def _apply_jets_mask(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        JETS 风格随机掩码：将信号分割为patch，随机保留一部分，其余置零。

        JETS 核心思想：上下文编码器只看部分patch，目标编码器看完整信号，
        迫使编码器从局部信息中学习能预测全局的强表征。

        Args:
            signal: (B, 1, L) — 原始ECG信号
        Returns:
            masked_signal: (B, 1, L) — 部分patch被置零的ECG
            token_mask: (B, 1, N_tokens) — True=可见token位置 (用于Token对齐)
        """
        if self.mask_ratio <= 0 or not self.training:
            # 推理时不做掩码 → 所有token可见
            dummy_mask = torch.ones(signal.size(0), 1, signal.size(-1)//16,
                                     device=signal.device, dtype=torch.bool)
            return signal, dummy_mask

        B, C, L = signal.shape
        patch_size = self.mask_patch_size
        num_patches = L // patch_size

        # 每个patch是否保留：随机保留 (1-mask_ratio) 的patch
        keep_prob = 1.0 - self.mask_ratio
        patch_mask = torch.rand(B, 1, num_patches, device=signal.device) < keep_prob

        # 将patch级别的mask展开到采样点级别 (B, 1, L)
        mask_expanded = patch_mask.repeat_interleave(patch_size, dim=-1)

        if mask_expanded.shape[-1] < L:
            pad_len = L - mask_expanded.shape[-1]
            mask_expanded = torch.cat(
                [mask_expanded, torch.ones(B, 1, pad_len, device=signal.device)],
                dim=-1
            )
        elif mask_expanded.shape[-1] > L:
            mask_expanded = mask_expanded[:, :, :L]

        # ★ Token mask: 在CNN输出维度上对应JETS掩码
        # CNN输出维度: 3000→188, 1000→63 (verified)
        n_tokens = ((L + 14) // 16)  # 近似公式, 覆盖padding
        # 用插值映射patch_mask到token级别
        token_mask = F.interpolate(
            patch_mask.float(), size=n_tokens, mode='nearest'
        ).bool()
        return signal * mask_expanded, token_mask

    def _make_token_block_mask(
        self, batch_size: int, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Create exact-ratio masks as unions of contiguous token blocks.

        The small mask is assembled on CPU and copied once. Reading a CUDA
        mask count inside the placement loop would otherwise synchronize the
        training stream repeatedly and leave avoidable gaps in GPU usage.
        """
        if num_tokens < 2:
            raise ValueError("Phase 1 token JEPA requires at least two tokens")
        target = int(round(num_tokens * self.phase1_mask_ratio))
        target = min(max(target, 1), num_tokens - 1)
        block = min(max(self.phase1_mask_block_tokens, 1), target)
        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)

        for batch_idx in range(batch_size):
            masked_count = 0
            attempts = 0
            while masked_count < target and attempts < num_tokens * 8:
                remaining = target - masked_count
                length = min(block, remaining)
                start = int(torch.randint(0, num_tokens - length + 1, (1,)))
                region = mask[batch_idx, start:start + length]
                newly_masked = int((~region).sum())
                region.fill_(True)
                masked_count += newly_masked
                attempts += 1

            missing = target - masked_count
            if missing > 0:
                available = (~mask[batch_idx]).nonzero(as_tuple=False).flatten()
                order = torch.randperm(available.numel())[:missing]
                mask[batch_idx, available[order]] = True
        return mask.to(device=device, non_blocking=True)

    @staticmethod
    def _masked_token_regression(
        prediction: torch.Tensor,
        target: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = F.normalize(prediction.float(), dim=-1)
        target = F.normalize(target.detach().float(), dim=-1)
        # Cosine distance is the dimension-independent form of normalized
        # squared error: 1 - cos(a, b) = ||a-b||^2 / 2. A per-dimension MSE
        # would shrink gradients in proportion to the latent width.
        per_token = 1.0 - (prediction * target).sum(dim=-1)
        return per_token[token_mask].mean()

    @staticmethod
    def _weighted_transport_regression(
        prediction: torch.Tensor,
        target: torch.Tensor,
        token_mask: torch.Tensor,
        available_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine token regression where the transport has a valid match."""
        prediction = F.normalize(prediction.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        per_token = 1.0 - (prediction * target).sum(dim=-1)
        valid = token_mask & (available_mass > 1e-6)
        if not valid.any():
            return prediction.sum() * 0.0
        return per_token[valid].mean()

    def set_phase2_progress(self, progress: float) -> None:
        """Set the transport blend in [0, 1] for the current epoch."""
        if self.pretrain_phase != 2:
            return
        self.phase2_progress = (
            min(max(float(progress), 0.0), 1.0)
            if self.phase2_transport_enabled
            else 0.0
        )

    def set_shared_private_progress(self, progress: float) -> None:
        """Ramp new private objectives independently from causal transport."""
        if (
            self.pretrain_phase != 2
            or not self.phase2_shared_private_enabled
        ):
            raise RuntimeError(
                "Shared-private progress requires an enabled Phase 2 model"
            )
        self.phase2_shared_private_progress = min(
            max(float(progress), 0.0), 1.0
        )

    def _phase2_v2_scores(
        self,
        ecg_tokens: torch.Tensor,
        ppg_tokens: torch.Tensor,
    ) -> dict:
        """Return cross-modal delay scores before Sinkhorn normalization."""
        batch_size, num_tokens, _ = ecg_tokens.shape
        offsets = self.phase2_delay_offsets
        num_bins = offsets.numel()
        source = torch.arange(
            num_tokens, device=ecg_tokens.device
        ).view(1, num_tokens, 1)
        target = source + offsets.view(1, 1, num_bins)
        valid_delay = target < num_tokens
        head = self.phase2_physio_transport(
            ecg_tokens, ppg_tokens, target
        )
        temperature = max(self.phase2_transport_temperature, 1e-4)
        scores = (
            self.phase2_v2_content_weight
            * head["content_scores"]
            + self.phase2_v2_global_delay_weight
            * head["global_delay_logits"]
            + self.phase2_v2_local_delay_weight
            * head["local_delay_logits"]
        ) / temperature
        scores = scores.masked_fill(~valid_delay, -1e4)
        row_score = torch.logsumexp(scores, dim=-1)
        valid_rows = valid_delay.any(dim=-1).expand(batch_size, -1)
        pair_score = (
            row_score - head["unmatched_logits"].float()
        ).masked_fill(~valid_rows, 0.0)
        pair_score = pair_score.sum(dim=-1) / valid_rows.sum(
            dim=-1
        ).clamp_min(1)
        return {
            **head,
            "scores": scores,
            "target": target,
            "valid_delay": valid_delay,
            "valid_rows": valid_rows,
            "pair_score": pair_score,
        }

    def _build_phase2_physio_transport(
        self,
        ecg_tokens: torch.Tensor,
        ppg_tokens: torch.Tensor,
    ) -> dict:
        """Build a positive-delay, dustbin-aware unbalanced Sinkhorn plan."""
        score_state = self._phase2_v2_scores(ecg_tokens, ppg_tokens)
        scores = score_state["scores"]
        target = score_state["target"]
        valid_delay = score_state["valid_delay"]
        batch_size, num_tokens, num_bins = scores.shape

        dense_scores = scores.new_full(
            (batch_size, num_tokens + 1, num_tokens + 1), -1e4
        )
        target_indices = target.clamp(max=num_tokens - 1).expand(
            batch_size, -1, -1
        )
        for bin_index, offset in enumerate(self.phase2_delay_offsets.tolist()):
            matchable = num_tokens - int(offset)
            if matchable <= 0:
                continue
            source_index = torch.arange(
                matchable, device=scores.device
            )
            dense_scores[
                :, source_index, source_index + int(offset)
            ] = scores[:, :matchable, bin_index]
        unmatched = score_state["unmatched_logits"].float()
        dense_scores[:, :num_tokens, num_tokens] = unmatched
        dense_scores[:, num_tokens, :num_tokens] = unmatched.mean(
            dim=1, keepdim=True
        )
        dense_scores[:, num_tokens, num_tokens] = 0.0

        epsilon = max(self.phase2_v2_sinkhorn_epsilon, 1e-4)
        log_kernel = dense_scores / epsilon
        log_mass = math.log(max(num_tokens, 1))
        log_a = log_kernel.new_zeros(num_tokens + 1)
        log_b = log_kernel.new_zeros(num_tokens + 1)
        log_a[-1] = log_mass
        log_b[-1] = log_mass
        relaxation = self.phase2_v2_sinkhorn_mass_reg / (
            self.phase2_v2_sinkhorn_mass_reg + epsilon
        )
        log_u = torch.zeros_like(log_kernel[..., 0])
        log_v = torch.zeros_like(log_kernel[..., 0])
        for _ in range(self.phase2_v2_sinkhorn_iters):
            log_u = relaxation * (
                log_a - torch.logsumexp(
                    log_kernel + log_v.unsqueeze(1), dim=2
                )
            )
            log_v = relaxation * (
                log_b - torch.logsumexp(
                    log_kernel + log_u.unsqueeze(2), dim=1
                )
            )
        log_plan = log_kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
        plan = torch.exp(log_plan.clamp(min=-40.0, max=20.0))

        real_support = (
            dense_scores[:, :num_tokens, :num_tokens] > -5e3
        )
        real_plan = (
            plan[:, :num_tokens, :num_tokens] * real_support
        )
        row_dustbin = plan[:, :num_tokens, num_tokens]
        row_total = real_plan.sum(dim=-1) + row_dustbin
        transport = real_plan / row_total.unsqueeze(-1).clamp_min(1e-8)
        unmatched_probability = row_dustbin / row_total.clamp_min(1e-8)
        match_mass = transport.sum(dim=-1)
        valid_rows = score_state["valid_rows"]

        forward_transport = real_plan / real_plan.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        forward_transport = forward_transport * valid_rows.unsqueeze(-1)
        column_mass = real_plan.sum(dim=1)
        valid_columns = column_mass > 1e-8
        reverse_transport = real_plan / column_mass.unsqueeze(1).clamp_min(1e-8)
        reverse_transport = reverse_transport * valid_columns.unsqueeze(1)

        delay_probabilities = scores.new_zeros(
            batch_size, num_tokens, num_bins
        )
        delay_probabilities = transport.gather(2, target_indices) * valid_delay
        conditional_delay = delay_probabilities / delay_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        offsets_float = self.phase2_delay_offsets.to(
            dtype=conditional_delay.dtype
        )
        expected_delay = (
            conditional_delay * offsets_float.view(1, 1, -1)
        ).sum(dim=-1)
        expanded_valid = valid_delay.expand(batch_size, -1, -1)
        return {
            "transport": transport,
            "forward_transport": forward_transport,
            "reverse_transport": reverse_transport,
            "delay_probabilities": delay_probabilities,
            "conditional_delay_probabilities": conditional_delay,
            "unmatched_probability": unmatched_probability,
            "match_mass": match_mass,
            "expected_delay": expected_delay,
            "valid_delay": expanded_valid,
            "valid_rows": valid_rows,
            "valid_columns": valid_columns,
            "pair_score": score_state["pair_score"],
            "sinkhorn_row_error": (
                plan.sum(dim=-1)[:, :num_tokens] - 1.0
            ).abs().mean(),
            "sinkhorn_column_error": (
                plan.sum(dim=1)[:, :num_tokens] - 1.0
            ).abs().mean(),
        }

    def _build_phase2_transport(
        self,
        ecg_tokens: torch.Tensor,
        ppg_tokens: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build the configured Transport policy with an unmatched dustbin."""
        if self.pretrain_phase != 2:
            raise RuntimeError("Phase 2 transport requested outside Phase 2")
        batch_size, num_tokens, _ = ecg_tokens.shape
        offsets = self.phase2_delay_offsets
        mode = self.phase2_transport_mode
        if mode == "physio_v2":
            if ppg_tokens is None:
                raise ValueError(
                    "physio_v2 Transport requires paired PPG tokens"
                )
            return self._build_phase2_physio_transport(
                ecg_tokens, ppg_tokens
            )
        if mode != "zero_delay" and num_tokens <= int(offsets.min().item()):
            raise ValueError(
                "Phase 2 token sequence is too short for the configured "
                "positive delay"
            )
        num_bins = offsets.numel()
        if mode == "zero_delay":
            identity = torch.eye(
                num_tokens,
                dtype=torch.float32,
                device=ecg_tokens.device,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            zeros = identity.new_zeros(batch_size, num_tokens)
            return {
                "transport": identity,
                "forward_transport": identity,
                "reverse_transport": identity,
                "delay_probabilities": identity.new_zeros(
                    batch_size, num_tokens, num_bins
                ),
                "conditional_delay_probabilities": identity.new_zeros(
                    batch_size, num_tokens, num_bins
                ),
                "unmatched_probability": zeros,
                "match_mass": torch.ones_like(zeros),
                "expected_delay": zeros,
                "valid_delay": torch.zeros(
                    batch_size,
                    num_tokens,
                    num_bins,
                    dtype=torch.bool,
                    device=ecg_tokens.device,
                ),
                "valid_rows": torch.ones(
                    batch_size,
                    num_tokens,
                    dtype=torch.bool,
                    device=ecg_tokens.device,
                ),
                "valid_columns": torch.ones(
                    batch_size,
                    num_tokens,
                    dtype=torch.bool,
                    device=ecg_tokens.device,
                ),
            }

        temperature = max(self.phase2_transport_temperature, 1e-4)
        logits = self.phase2_delay_head(ecg_tokens).float() / temperature

        source = torch.arange(num_tokens, device=ecg_tokens.device).view(1, num_tokens, 1)
        target = source + offsets.view(1, 1, num_bins)
        valid_delay = target < num_tokens
        delay_logits = logits[..., :num_bins].masked_fill(~valid_delay, -1e4)

        if mode == "fixed_prior":
            prior_index = int(
                torch.argmin(
                    (
                        offsets.float()
                        - float(self.phase2_delay_prior_tokens)
                    ).abs()
                ).item()
            )
            selected_valid = valid_delay[..., prior_index].expand(
                batch_size, -1
            )
            delay_logits = delay_logits.new_full(
                (batch_size, num_tokens, num_bins), -1e4
            )
            delay_logits[..., prior_index] = torch.where(
                selected_valid,
                delay_logits.new_zeros(()),
                delay_logits.new_full((), -1e4),
            )
            unmatched_logits = torch.where(
                selected_valid,
                delay_logits.new_full(selected_valid.shape, -1e4),
                delay_logits.new_zeros(selected_valid.shape),
            ).unsqueeze(-1)
        else:
            unmatched_logits = logits[..., -1:]

        probabilities = F.softmax(
            torch.cat([delay_logits, unmatched_logits], dim=-1),
            dim=-1,
        )
        base_delay_probabilities = probabilities[..., :num_bins] * valid_delay
        unmatched_probability = probabilities[..., -1]
        match_mass = base_delay_probabilities.sum(dim=-1)
        expanded_valid_delay = valid_delay.expand(batch_size, -1, -1)

        # Condition on a token being matched using a separate softmax. Dividing
        # by match_mass is algebraically equivalent, but creates 1/mass
        # gradients when the dustbin probability is close to one. Those
        # gradients can overflow under AMP even while the forward loss is
        # finite.
        conditional_delay = F.softmax(delay_logits, dim=-1) * valid_delay
        if mode == "static_delay":
            conditional_delay = conditional_delay.mean(
                dim=1, keepdim=True
            ).expand_as(conditional_delay)
        elif mode == "token_shuffled":
            conditional_delay = torch.roll(
                conditional_delay,
                shifts=max(1, num_tokens // 4),
                dims=1,
            )
        if mode in {"static_delay", "token_shuffled"}:
            conditional_delay = conditional_delay * valid_delay
            conditional_delay = conditional_delay / conditional_delay.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        policy_valid = expanded_valid_delay & (conditional_delay > 0)
        valid_rows = policy_valid.any(dim=-1)
        delay_probabilities = conditional_delay * match_mass.unsqueeze(-1)

        target_indices = target.clamp(max=max(num_tokens - 1, 0)).expand(
            batch_size, -1, -1
        )
        transport = delay_probabilities.new_zeros(
            batch_size, num_tokens, num_tokens
        )
        transport.scatter_add_(2, target_indices, delay_probabilities)
        forward_transport = conditional_delay.new_zeros(
            batch_size, num_tokens, num_tokens
        )
        forward_transport.scatter_add_(2, target_indices, conditional_delay)

        # Normalize the reverse direction with a column-wise masked softmax,
        # avoiding another division by a potentially tiny column mass.
        dummy_target = torch.full_like(target, num_tokens)
        dense_indices = torch.where(
            policy_valid,
            target.expand(batch_size, -1, -1),
            dummy_target.expand(batch_size, -1, -1),
        )
        policy_logits = conditional_delay.clamp_min(1e-8).log().masked_fill(
            ~policy_valid, -1e4
        )
        reverse_logits = delay_logits.new_full(
            (batch_size, num_tokens, num_tokens + 1), -1e4
        ).scatter(2, dense_indices, policy_logits)
        reverse_logits = reverse_logits[..., :num_tokens]
        reverse_valid = torch.zeros(
            (batch_size, num_tokens, num_tokens + 1),
            dtype=torch.bool,
            device=ecg_tokens.device,
        ).scatter(2, dense_indices, policy_valid)
        reverse_valid = reverse_valid[..., :num_tokens]
        reverse_transport = F.softmax(reverse_logits, dim=1) * reverse_valid
        valid_columns = reverse_valid.any(dim=1)

        offsets_float = offsets.to(dtype=delay_probabilities.dtype)
        expected_delay = (
            conditional_delay * offsets_float.view(1, 1, -1)
        ).sum(dim=-1)
        return {
            "transport": transport,
            "forward_transport": forward_transport,
            "reverse_transport": reverse_transport,
            "delay_probabilities": delay_probabilities,
            "conditional_delay_probabilities": conditional_delay,
            "unmatched_probability": unmatched_probability,
            "match_mass": match_mass,
            "expected_delay": expected_delay,
            "valid_delay": policy_valid,
            "valid_rows": valid_rows,
            "valid_columns": valid_columns,
        }

    def _phase2_effective_regularizer_weights(self) -> dict:
        """Return only the constraint weights active in this ablation mode."""
        weights = {
            "delay_prior": self.phase2_delay_prior_weight,
            "monotonic": self.phase2_monotonic_weight,
            "delay_smoothness": self.phase2_delay_smoothness_weight,
            "match_mass": self.phase2_match_mass_weight,
        }
        if self.phase2_transport_mode == "no_monotonic":
            weights["monotonic"] = 0.0
        elif self.phase2_transport_mode in {"fixed_prior", "zero_delay"}:
            # These policies are fixed by construction; policy regularizers
            # would add constants without training a delay distribution.
            weights = {name: 0.0 for name in weights}
        return weights

    def _phase2_transport_regularizers(self, state: dict) -> dict:
        """Positive-delay prior, monotonicity, smoothness, and mass control."""
        probabilities = state["delay_probabilities"]
        match_mass = state["match_mass"]
        expected_delay = state["expected_delay"]
        valid_delay = state["valid_delay"]
        valid_rows = state["valid_rows"]
        offsets = self.phase2_delay_offsets.to(probabilities.dtype)

        conditional = state["conditional_delay_probabilities"]
        prior_scale = max(float(offsets.numel()) / 3.0, 1.0)
        prior = torch.exp(
            -0.5 * ((offsets - self.phase2_delay_prior_tokens) / prior_scale).square()
        ).view(1, 1, -1)
        prior = prior * valid_delay
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        prior_kl_per_row = (
            conditional
            * (
                conditional.clamp_min(1e-8).log()
                - prior.clamp_min(1e-8).log()
            )
        ).sum(dim=-1)
        delay_prior = prior_kl_per_row[valid_rows].mean()

        positions = torch.arange(
            expected_delay.size(1),
            device=expected_delay.device,
            dtype=expected_delay.dtype,
        ).view(1, -1)
        expected_target = positions + expected_delay
        pair_valid = valid_rows[:, :-1] & valid_rows[:, 1:]
        monotonic_per_pair = F.relu(
            expected_target[:, :-1] - expected_target[:, 1:]
        )
        monotonic = (
            monotonic_per_pair[pair_valid].mean()
            if pair_valid.any()
            else monotonic_per_pair.sum() * 0.0
        )

        normalized_delay = expected_delay / offsets.max().clamp_min(1.0)
        smoothness_per_pair = F.smooth_l1_loss(
            normalized_delay[:, 1:],
            normalized_delay[:, :-1],
            reduction="none",
        )
        smoothness = (
            smoothness_per_pair[pair_valid].mean()
            if pair_valid.any()
            else smoothness_per_pair.sum() * 0.0
        )
        match_mass_loss = (
            match_mass[valid_rows].mean() - self.phase2_target_match_mass
        ).square()

        conditional_entropy = -(
            conditional * conditional.clamp_min(1e-8).log()
        ).sum(dim=-1)
        valid_expected = expected_delay[valid_rows]
        return {
            "delay_prior_loss": delay_prior,
            "monotonic_loss": monotonic,
            "delay_smoothness_loss": smoothness,
            "match_mass_loss": match_mass_loss,
            "delay_mean_ms": valid_expected.mean() * self.phase2_token_ms,
            "delay_std_ms": valid_expected.std(unbiased=False) * self.phase2_token_ms,
            "transport_entropy": conditional_entropy[valid_rows].mean(),
            "matched_mass": match_mass[valid_rows].mean(),
            "minimum_matched_mass": match_mass[valid_rows].min(),
            "unmatched_mass": state["unmatched_probability"].mean(),
        }

    @staticmethod
    def _variance_covariance_regularization(embeddings, target_std: float):
        """VICReg-style anti-collapse regularization across independent samples."""
        variance_losses = []
        covariance_losses = []
        for embedding in embeddings:
            values = embedding.float()
            std = torch.sqrt(values.var(dim=0, unbiased=False) + 1e-4)
            variance_losses.append(F.relu(float(target_std) - std).mean())
            if values.size(0) > 1:
                centered = values - values.mean(dim=0, keepdim=True)
                covariance = centered.T @ centered / (values.size(0) - 1)
                off_diagonal = covariance - torch.diag_embed(
                    torch.diagonal(covariance)
                )
                covariance_losses.append(
                    off_diagonal.square().sum() / max(1, values.size(1))
                )
        variance_loss = torch.stack(variance_losses).mean()
        covariance_loss = (
            torch.stack(covariance_losses).mean()
            if covariance_losses
            else variance_loss * 0.0
        )
        return variance_loss, covariance_loss

    @staticmethod
    def _shared_private_orthogonality(
        shared_tokens: torch.Tensor,
        private_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize sample-level cross-correlation between the two views."""
        shared = shared_tokens.float().mean(dim=1)
        private = private_tokens.float().mean(dim=1)
        if shared.size(0) < 2:
            return shared.sum() * 0.0

        shared = shared - shared.mean(dim=0, keepdim=True)
        private = private - private.mean(dim=0, keepdim=True)
        shared = shared / torch.sqrt(
            shared.var(dim=0, unbiased=False, keepdim=True) + 1e-4
        )
        private = private / torch.sqrt(
            private.var(dim=0, unbiased=False, keepdim=True) + 1e-4
        )
        cross_correlation = shared.transpose(0, 1) @ private
        cross_correlation = cross_correlation / max(1, shared.size(0) - 1)
        return cross_correlation.square().mean()

    @staticmethod
    def _add_embedding_diagnostics(
        info: dict,
        embeddings,
        collect_diagnostics: bool,
    ) -> None:
        for prefix, embedding in embeddings:
            detached = embedding.detach().float()
            variances = detached.var(dim=0, unbiased=False)
            info[f"{prefix}_std"] = variances.sqrt().mean().item()
            info[f"{prefix}_collapsed_fraction"] = (
                variances < 1e-4
            ).float().mean().item()
            if collect_diagnostics and detached.size(0) > 1:
                centered = detached - detached.mean(dim=0, keepdim=True)
                covariance = centered.T @ centered / (detached.size(0) - 1)
                off_diagonal = covariance - torch.diag_embed(
                    torch.diagonal(covariance)
                )
                info[f"{prefix}_cov_offdiag_rms"] = (
                    off_diagonal.square().mean().sqrt().item()
                )

    def _compute_phase1_loss(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        ecg_stats: Optional[torch.Tensor],
        collect_diagnostics: bool,
    ) -> Tuple[torch.Tensor, dict]:
        """Bidirectional masked-token JEPA with same-modality EMA teachers."""
        ecg_input_tokens = self.context_encoder.tokenize(ecg)
        ppg_input_tokens = self.ppg_encoder.tokenize(ppg)
        num_tokens = min(ecg_input_tokens.size(1), ppg_input_tokens.size(1))
        ecg_input_tokens = ecg_input_tokens[:, :num_tokens]
        ppg_input_tokens = ppg_input_tokens[:, :num_tokens]
        token_mask = self._make_token_block_mask(
            ecg.size(0), num_tokens, ecg.device
        )

        ecg_pooled, ecg_tokens = self.context_encoder.encode_tokens(
            ecg_input_tokens,
            return_all=True,
            token_mask=token_mask,
            mask_token=self.ecg_mask_token,
        )
        ppg_pooled, ppg_tokens = self.ppg_encoder.encode_tokens(
            ppg_input_tokens,
            return_all=True,
            token_mask=token_mask,
            mask_token=self.ppg_mask_token,
        )

        with torch.no_grad():
            ecg_teacher_pooled, ecg_teacher_tokens = self.context_teacher(
                ecg, return_all=True
            )
            ppg_teacher_pooled, ppg_teacher_tokens = self.target_encoder(
                ppg, return_all=True
            )
            ecg_teacher_tokens = self.ecg_teacher_proj(
                ecg_teacher_tokens[:, :num_tokens]
            )
            ppg_teacher_tokens = self.target_proj(
                ppg_teacher_tokens[:, :num_tokens]
            )

        ecg_online_tokens = self.ecg_token_proj(ecg_tokens)
        ppg_online_tokens = self.ppg_token_proj(ppg_tokens)
        ppg_prediction = self.ecg_to_ppg_predictor(ecg_online_tokens)
        ecg_to_ppg = self._masked_token_regression(
            ppg_prediction, ppg_teacher_tokens, token_mask
        )

        if self.phase1_bidirectional:
            ecg_prediction = self.ppg_to_ecg_predictor(ppg_online_tokens)
            ppg_to_ecg = self._masked_token_regression(
                ecg_prediction, ecg_teacher_tokens, token_mask
            )
            token_jepa = 0.5 * (ecg_to_ppg + ppg_to_ecg)
            prediction_std = 0.5 * (
                ppg_prediction[token_mask].detach().float().std(unbiased=False)
                + ecg_prediction[token_mask].detach().float().std(unbiased=False)
            )
        else:
            ppg_to_ecg = torch.zeros_like(ecg_to_ppg)
            token_jepa = ecg_to_ppg
            prediction_std = ppg_prediction[token_mask].detach().float().std(
                unbiased=False
            )

        total_loss = self.phase1_token_loss_weight * token_jepa
        info = {
            "jepa": token_jepa.item(),
            "ecg_to_ppg_token": ecg_to_ppg.item(),
            "ppg_to_ecg_token": ppg_to_ecg.item(),
            "masked_fraction": token_mask.float().mean().item(),
            "prediction_std": prediction_std.item(),
            "token_align": 0.0,
        }
        self._add_embedding_diagnostics(
            info,
            (
                ("context", ecg_pooled),
                ("target", ppg_pooled),
                ("ecg_teacher", ecg_teacher_pooled),
                ("ppg_teacher", ppg_teacher_pooled),
            ),
            collect_diagnostics,
        )

        if self.use_stats_loss and ecg_stats is not None:
            stats_loss, stats_info = self._compute_stats_loss(
                ecg_pooled, ecg_stats
            )
            total_loss = total_loss + self.stats_loss_weight * stats_loss
            info.update(stats_info)

        info["total_loss"] = total_loss.item()
        return total_loss, info

    def _compute_phase2_loss(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        ecg_stats: Optional[torch.Tensor],
        collect_diagnostics: bool,
        return_components: bool = False,
        pat_target_ms: Optional[torch.Tensor] = None,
        pat_confidence: Optional[torch.Tensor] = None,
    ):
        """Phase 1 token JEPA extended with causal monotonic transport."""
        ecg_input_tokens = self.context_encoder.tokenize(ecg)
        ppg_input_tokens = self.ppg_encoder.tokenize(ppg)
        num_tokens = min(ecg_input_tokens.size(1), ppg_input_tokens.size(1))
        ecg_input_tokens = ecg_input_tokens[:, :num_tokens]
        ppg_input_tokens = ppg_input_tokens[:, :num_tokens]
        token_mask = self._make_token_block_mask(
            ecg.size(0), num_tokens, ecg.device
        )

        ecg_pooled, ecg_tokens = self.context_encoder.encode_tokens(
            ecg_input_tokens,
            return_all=True,
            token_mask=token_mask,
            mask_token=self.ecg_mask_token,
        )
        ppg_pooled, ppg_tokens = self.ppg_encoder.encode_tokens(
            ppg_input_tokens,
            return_all=True,
            token_mask=token_mask,
            mask_token=self.ppg_mask_token,
        )

        with torch.no_grad():
            ecg_teacher_pooled, ecg_teacher_tokens = self.context_teacher(
                ecg, return_all=True
            )
            ppg_teacher_pooled, ppg_teacher_tokens = self.target_encoder(
                ppg, return_all=True
            )
            ecg_teacher_tokens = self.ecg_teacher_proj(
                ecg_teacher_tokens[:, :num_tokens]
            )
            ppg_teacher_tokens = self.target_proj(
                ppg_teacher_tokens[:, :num_tokens]
            )

        ecg_online_tokens = self.ecg_token_proj(ecg_tokens)
        ppg_online_tokens = self.ppg_token_proj(ppg_tokens)
        ecg_shared_tokens = ecg_online_tokens
        ppg_shared_tokens = ppg_online_tokens
        ecg_teacher_shared = ecg_teacher_tokens
        ppg_teacher_shared = ppg_teacher_tokens
        ecg_private_tokens = None
        ppg_private_tokens = None
        ecg_teacher_private = None
        ppg_teacher_private = None
        if self.phase2_shared_private_enabled:
            ecg_shared_tokens, ecg_private_tokens = self.ecg_shared_private(
                ecg_online_tokens
            )
            ppg_shared_tokens, ppg_private_tokens = self.ppg_shared_private(
                ppg_online_tokens
            )
            with torch.no_grad():
                (
                    ecg_teacher_shared,
                    ecg_teacher_private,
                ) = self.ecg_teacher_shared_private(ecg_teacher_tokens)
                (
                    ppg_teacher_shared,
                    ppg_teacher_private,
                ) = self.ppg_teacher_shared_private(ppg_teacher_tokens)

        # Cross-modal prediction and causal transport are deliberately
        # restricted to the shared representation.
        ppg_prediction = self.ecg_to_ppg_predictor(ecg_shared_tokens)
        ecg_prediction = self.ppg_to_ecg_predictor(ppg_shared_tokens)

        direct_ecg_to_ppg = self._masked_token_regression(
            ppg_prediction, ppg_teacher_shared, token_mask
        )
        direct_ppg_to_ecg = self._masked_token_regression(
            ecg_prediction, ecg_teacher_shared, token_mask
        )
        direct_jepa = 0.5 * (direct_ecg_to_ppg + direct_ppg_to_ecg)

        zero = direct_jepa.new_zeros(())
        if self.phase2_transport_enabled:
            transport_state = self._build_phase2_transport(
                ecg_shared_tokens,
                ppg_teacher_shared
                if self.phase2_transport_mode == "physio_v2"
                else None,
            )
            forward_transport = transport_state["forward_transport"]
            reverse_transport = transport_state["reverse_transport"]
            transported_ppg = torch.bmm(
                forward_transport, ppg_teacher_shared.float()
            )
            transported_ecg = torch.bmm(
                reverse_transport.transpose(1, 2),
                ecg_teacher_shared.float(),
            )

            transport_ecg_to_ppg = self._weighted_transport_regression(
                ppg_prediction,
                transported_ppg,
                token_mask,
                transport_state["valid_rows"],
            )
            transport_ppg_to_ecg = self._weighted_transport_regression(
                ecg_prediction,
                transported_ecg,
                token_mask,
                transport_state["valid_columns"],
            )
            transport_jepa = 0.5 * (
                transport_ecg_to_ppg + transport_ppg_to_ecg
            )
            regularizers = self._phase2_transport_regularizers(
                transport_state
            )
            counterfactual_loss = zero
            counterfactual_accuracy = zero
            if self.phase2_transport_mode == "physio_v2":
                negative_ppg = (
                    torch.roll(ppg_teacher_shared, shifts=1, dims=0)
                    if ppg_teacher_shared.size(0) > 1
                    else torch.flip(ppg_teacher_shared, dims=(1,))
                )
                negative_score = self._phase2_v2_scores(
                    ecg_shared_tokens, negative_ppg
                )["pair_score"]
                positive_score = transport_state["pair_score"]
                counterfactual_loss = F.relu(
                    self.phase2_counterfactual_margin
                    - positive_score
                    + negative_score
                ).mean()
                counterfactual_accuracy = (
                    positive_score > negative_score
                ).float().mean()
            progress = self.phase2_progress
            token_jepa = (
                (1.0 - progress) * direct_jepa
                + progress
                * self.phase2_transport_loss_weight
                * transport_jepa
            )
        else:
            transport_ecg_to_ppg = zero
            transport_ppg_to_ecg = zero
            transport_jepa = zero
            progress = 0.0
            counterfactual_loss = zero
            counterfactual_accuracy = zero
            token_jepa = direct_jepa
            regularizers = {
                "delay_prior_loss": zero,
                "monotonic_loss": zero,
                "delay_smoothness_loss": zero,
                "match_mass_loss": zero,
                "delay_mean_ms": zero,
                "delay_std_ms": zero,
                "transport_entropy": zero,
                "matched_mass": zero,
                "minimum_matched_mass": zero,
                "unmatched_mass": zero,
            }

        pat_weak_loss = token_jepa.new_zeros(())
        pat_valid_fraction = token_jepa.new_zeros(())
        if (
            self.phase2_transport_enabled
            and self.phase2_transport_mode == "physio_v2"
            and pat_target_ms is not None
        ):
            predicted_pat = (
                transport_state["expected_delay"]
                * transport_state["valid_rows"]
            ).sum(dim=1) / transport_state["valid_rows"].sum(
                dim=1
            ).clamp_min(1)
            predicted_pat = predicted_pat * self.phase2_token_ms
            target_pat = pat_target_ms.to(
                device=predicted_pat.device,
                dtype=predicted_pat.dtype,
            ).reshape(-1)
            valid_pat = torch.isfinite(target_pat)
            valid_pat = valid_pat & (
                target_pat >= self.phase2_delay_offsets.min()
                * self.phase2_token_ms
            )
            valid_pat = valid_pat & (
                target_pat <= self.phase2_delay_offsets.max()
                * self.phase2_token_ms
            )
            if valid_pat.any():
                per_sample_pat = F.smooth_l1_loss(
                    predicted_pat[valid_pat] / 1000.0,
                    target_pat[valid_pat] / 1000.0,
                    reduction="none",
                )
                if pat_confidence is not None:
                    confidence = pat_confidence.to(
                        device=predicted_pat.device,
                        dtype=predicted_pat.dtype,
                    ).reshape(-1)[valid_pat].clamp(0.0, 1.0)
                    pat_weak_loss = (
                        per_sample_pat * confidence
                    ).sum() / confidence.sum().clamp_min(1e-6)
                else:
                    pat_weak_loss = per_sample_pat.mean()
                pat_valid_fraction = valid_pat.float().mean()

        private_reconstruction = token_jepa.new_zeros(())
        ecg_private_reconstruction = token_jepa.new_zeros(())
        ppg_private_reconstruction = token_jepa.new_zeros(())
        shared_private_orthogonality = token_jepa.new_zeros(())
        regularization_embeddings = [
            ecg_pooled,
            ppg_pooled,
            ecg_shared_tokens.mean(dim=1),
            ppg_shared_tokens.mean(dim=1),
        ]
        if self.phase2_shared_private_enabled:
            ecg_private_prediction = self.ecg_private_predictor(
                ecg_private_tokens
            )
            ppg_private_prediction = self.ppg_private_predictor(
                ppg_private_tokens
            )
            ecg_private_reconstruction = self._masked_token_regression(
                ecg_private_prediction,
                ecg_teacher_private,
                token_mask,
            )
            ppg_private_reconstruction = self._masked_token_regression(
                ppg_private_prediction,
                ppg_teacher_private,
                token_mask,
            )
            private_reconstruction = 0.5 * (
                ecg_private_reconstruction + ppg_private_reconstruction
            )
            shared_private_orthogonality = 0.5 * (
                self._shared_private_orthogonality(
                    ecg_shared_tokens, ecg_private_tokens
                )
                + self._shared_private_orthogonality(
                    ppg_shared_tokens, ppg_private_tokens
                )
            )
            regularization_embeddings.extend(
                (
                    ecg_private_tokens.mean(dim=1),
                    ppg_private_tokens.mean(dim=1),
                )
            )

        variance_loss, covariance_loss = self._variance_covariance_regularization(
            regularization_embeddings,
            self.phase2_target_std,
        )
        total_loss = self.phase1_token_loss_weight * token_jepa
        total_loss = total_loss + self.phase2_variance_weight * variance_loss
        total_loss = total_loss + self.phase2_covariance_weight * covariance_loss
        constraint_weights = self._phase2_effective_regularizer_weights()
        total_loss = total_loss + progress * (
            constraint_weights["delay_prior"]
            * regularizers["delay_prior_loss"]
            + constraint_weights["monotonic"]
            * regularizers["monotonic_loss"]
            + constraint_weights["delay_smoothness"]
            * regularizers["delay_smoothness_loss"]
            + constraint_weights["match_mass"]
            * regularizers["match_mass_loss"]
            + self.phase2_counterfactual_weight * counterfactual_loss
            + self.phase2_pat_weak_weight * pat_weak_loss
        )
        shared_private_progress = (
            self.phase2_shared_private_progress
            if self.phase2_shared_private_enabled
            else 0.0
        )
        total_loss = total_loss + shared_private_progress * (
            self.phase2_private_loss_weight * private_reconstruction
            + self.phase2_orthogonality_weight * shared_private_orthogonality
        )

        prediction_std = 0.5 * (
            ppg_prediction[token_mask].detach().float().std(unbiased=False)
            + ecg_prediction[token_mask].detach().float().std(unbiased=False)
        )
        info = {
            "jepa": token_jepa.item(),
            "direct_token_jepa": direct_jepa.item(),
            "transport_token_jepa": transport_jepa.item(),
            "ecg_to_ppg_token": transport_ecg_to_ppg.item(),
            "ppg_to_ecg_token": transport_ppg_to_ecg.item(),
            "masked_fraction": token_mask.float().mean().item(),
            "prediction_std": prediction_std.item(),
            "phase2_progress": progress,
            "phase2_transport_enabled": self.phase2_transport_enabled,
            "phase2_transport_mode": self.phase2_transport_mode,
            "shared_private_progress": shared_private_progress,
            "private_reconstruction": private_reconstruction.item(),
            "ecg_private_reconstruction": ecg_private_reconstruction.item(),
            "ppg_private_reconstruction": ppg_private_reconstruction.item(),
            "shared_private_orthogonality": (
                shared_private_orthogonality.item()
            ),
            "delay_prior": regularizers["delay_prior_loss"].item(),
            "monotonic": regularizers["monotonic_loss"].item(),
            "delay_smoothness": regularizers["delay_smoothness_loss"].item(),
            "match_mass_loss": regularizers["match_mass_loss"].item(),
            "variance": variance_loss.item(),
            "covariance": covariance_loss.item(),
            "delay_mean_ms": regularizers["delay_mean_ms"].item(),
            "delay_std_ms": regularizers["delay_std_ms"].item(),
            "transport_entropy": regularizers["transport_entropy"].item(),
            "matched_mass": regularizers["matched_mass"].item(),
            "minimum_matched_mass": regularizers[
                "minimum_matched_mass"
            ].item(),
            "unmatched_mass": regularizers["unmatched_mass"].item(),
            "counterfactual_loss": counterfactual_loss.item(),
            "counterfactual_accuracy": counterfactual_accuracy.item(),
            "pat_weak_loss": pat_weak_loss.item(),
            "pat_valid_fraction": pat_valid_fraction.item(),
            "token_align": 0.0,
        }
        if (
            self.phase2_transport_enabled
            and self.phase2_transport_mode == "physio_v2"
        ):
            info["sinkhorn_row_error"] = transport_state[
                "sinkhorn_row_error"
            ].item()
            info["sinkhorn_column_error"] = transport_state[
                "sinkhorn_column_error"
            ].item()
        self._add_embedding_diagnostics(
            info,
            (
                ("context", ecg_pooled),
                ("target", ppg_pooled),
                ("ecg_teacher", ecg_teacher_pooled),
                ("ppg_teacher", ppg_teacher_pooled),
            ),
            collect_diagnostics,
        )
        if self.phase2_shared_private_enabled:
            self._add_embedding_diagnostics(
                info,
                (
                    ("ecg_shared", ecg_shared_tokens.mean(dim=1)),
                    ("ppg_shared", ppg_shared_tokens.mean(dim=1)),
                    ("ecg_private", ecg_private_tokens.mean(dim=1)),
                    ("ppg_private", ppg_private_tokens.mean(dim=1)),
                ),
                collect_diagnostics,
            )

        stats_loss = total_loss.new_zeros(())
        if self.use_stats_loss and ecg_stats is not None:
            stats_loss, stats_info = self._compute_stats_loss(
                ecg_pooled, ecg_stats
            )
            total_loss = total_loss + self.stats_loss_weight * stats_loss
            info.update(stats_info)

        info["total_loss"] = total_loss.item()
        if not return_components:
            return total_loss, info
        components = {
            "direct_token_jepa": direct_jepa,
            "transport_token_jepa": transport_jepa,
            "token_jepa": token_jepa,
            "delay_prior": regularizers["delay_prior_loss"],
            "monotonic": regularizers["monotonic_loss"],
            "delay_smoothness": regularizers["delay_smoothness_loss"],
            "match_mass": regularizers["match_mass_loss"],
            "variance": variance_loss,
            "covariance": covariance_loss,
            "private_reconstruction": private_reconstruction,
            "shared_private_orthogonality": shared_private_orthogonality,
            "counterfactual": counterfactual_loss,
            "pat_weak": pat_weak_loss,
            "stats": stats_loss,
            "total": total_loss,
        }
        return total_loss, info, components

    def compute_loss(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        ecg_stats: Optional[torch.Tensor] = None,
        collect_diagnostics: bool = False,
        return_components: bool = False,
        pat_target_ms: Optional[torch.Tensor] = None,
        pat_confidence: Optional[torch.Tensor] = None,
    ):
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
        if self.pretrain_phase == 2:
            return self._compute_phase2_loss(
                ecg,
                ppg,
                ecg_stats,
                collect_diagnostics,
                return_components,
                pat_target_ms,
                pat_confidence,
            )
        if return_components:
            raise ValueError("return_components is currently supported only in Phase 2")
        if self.pretrain_phase == 1:
            return self._compute_phase1_loss(
                ecg, ppg, ecg_stats, collect_diagnostics
            )

        B = ecg.size(0)

        # ★ JETS 式掩码：随机丢弃 ~70% patch，强制编码器从局部学习全局表征
        ecg_masked, token_mask = self._apply_jets_mask(ecg)

        # ★ 一次前向同时拿 pooled embedding + token 序列 (避免二次前向导致 OOM)
        need_tokens = self.use_token_align
        context_out = self.forward_context(ecg_masked, return_tokens=need_tokens)
        if need_tokens:
            context_embed, ctx_tokens = context_out
        else:
            context_embed = context_out
            ctx_tokens = None
        target_out = self.forward_target(ppg, return_tokens=need_tokens)
        if need_tokens:
            target_embed, tgt_tokens = target_out
        else:
            target_embed = target_out
            tgt_tokens = None

        # 1. JEPA prediction loss (ECG → PPG)
        jepa_loss, jepa_info = self._compute_jepa_loss(
            ecg, ppg, context_embed, target_embed
        )
        total_loss = jepa_loss
        info = dict(jepa_info)

        # Cheap collapse indicators are logged for both train and validation.
        for prefix, embedding in (
            ("context", context_embed),
            ("target", target_embed),
        ):
            detached = embedding.detach().float()
            variances = detached.var(dim=0, unbiased=False)
            info[f"{prefix}_std"] = variances.sqrt().mean().item()
            info[f"{prefix}_collapsed_fraction"] = (
                variances < 1e-4
            ).float().mean().item()
            if collect_diagnostics and detached.size(0) > 1:
                centered = detached - detached.mean(dim=0, keepdim=True)
                covariance = centered.T @ centered / (detached.size(0) - 1)
                off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
                info[f"{prefix}_cov_offdiag_rms"] = (
                    off_diagonal.square().mean().sqrt().item()
                )

        # 2. Statistics prediction loss (auxiliary)
        if self.use_stats_loss and ecg_stats is not None:
            stats_loss, stats_info = self._compute_stats_loss(
                context_embed, ecg_stats
            )
            total_loss = total_loss + self.stats_loss_weight * stats_loss
            info.update(stats_info)

        # 3. ★ Soft-DTW 弹性 Token 对齐 (用缓存的 token, 无需二次前向)
        if self.use_token_align and ctx_tokens is not None and tgt_tokens is not None:
            token_loss, token_info = self._compute_token_align_loss(
                ecg, ppg, token_mask=token_mask,
                align_window=self.align_window,
                soft_temperature=0.1,
                cached_ctx_tokens=ctx_tokens,
                cached_tgt_tokens=tgt_tokens,
            )
            total_loss = total_loss + self.token_align_weight * token_loss
            info.update(token_info)

        info["total_loss"] = total_loss.item()
        return total_loss, info
