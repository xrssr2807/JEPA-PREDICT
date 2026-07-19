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
                self.phase2_progress = 0.0

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
        self.phase2_progress = min(max(float(progress), 0.0), 1.0)

    def _build_phase2_transport(self, ecg_tokens: torch.Tensor) -> dict:
        """Build a causal banded transport with an unmatched dustbin."""
        if self.pretrain_phase != 2:
            raise RuntimeError("Phase 2 transport requested outside Phase 2")
        batch_size, num_tokens, _ = ecg_tokens.shape
        offsets = self.phase2_delay_offsets
        if num_tokens <= int(offsets.min().item()):
            raise ValueError(
                "Phase 2 token sequence is too short for the configured "
                "positive delay"
            )
        num_bins = offsets.numel()
        temperature = max(self.phase2_transport_temperature, 1e-4)
        logits = self.phase2_delay_head(ecg_tokens).float() / temperature

        source = torch.arange(num_tokens, device=ecg_tokens.device).view(1, num_tokens, 1)
        target = source + offsets.view(1, 1, num_bins)
        valid_delay = target < num_tokens
        delay_logits = logits[..., :num_bins].masked_fill(~valid_delay, -1e4)
        logits = torch.cat([delay_logits, logits[..., -1:]], dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        delay_probabilities = probabilities[..., :num_bins] * valid_delay
        unmatched_probability = probabilities[..., -1]
        match_mass = delay_probabilities.sum(dim=-1)
        valid_rows = valid_delay.any(dim=-1).expand(batch_size, -1)

        # Condition on a token being matched using a separate softmax. Dividing
        # by match_mass is algebraically equivalent, but creates 1/mass
        # gradients when the dustbin probability is close to one. Those
        # gradients can overflow under AMP even while the forward loss is
        # finite.
        conditional_delay = F.softmax(delay_logits, dim=-1) * valid_delay

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
        dense_indices = torch.where(valid_delay, target, dummy_target).expand(
            batch_size, -1, -1
        )
        reverse_logits = delay_logits.new_full(
            (batch_size, num_tokens, num_tokens + 1), -1e4
        ).scatter(2, dense_indices, delay_logits)
        reverse_logits = reverse_logits[..., :num_tokens]
        reverse_valid = torch.zeros(
            (1, num_tokens, num_tokens + 1),
            dtype=torch.bool,
            device=ecg_tokens.device,
        ).scatter(2, torch.where(valid_delay, target, dummy_target), valid_delay)
        reverse_valid = reverse_valid[..., :num_tokens].expand(batch_size, -1, -1)
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
            "valid_delay": valid_delay.expand(batch_size, -1, -1),
            "valid_rows": valid_rows,
            "valid_columns": valid_columns,
        }

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
    ) -> Tuple[torch.Tensor, dict]:
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
        ppg_prediction = self.ecg_to_ppg_predictor(ecg_online_tokens)
        ecg_prediction = self.ppg_to_ecg_predictor(ppg_online_tokens)

        direct_ecg_to_ppg = self._masked_token_regression(
            ppg_prediction, ppg_teacher_tokens, token_mask
        )
        direct_ppg_to_ecg = self._masked_token_regression(
            ecg_prediction, ecg_teacher_tokens, token_mask
        )
        direct_jepa = 0.5 * (direct_ecg_to_ppg + direct_ppg_to_ecg)

        transport_state = self._build_phase2_transport(ecg_online_tokens)
        forward_transport = transport_state["forward_transport"]
        reverse_transport = transport_state["reverse_transport"]
        transported_ppg = torch.bmm(
            forward_transport, ppg_teacher_tokens.float()
        )
        transported_ecg = torch.bmm(
            reverse_transport.transpose(1, 2), ecg_teacher_tokens.float()
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

        progress = self.phase2_progress
        token_jepa = (
            (1.0 - progress) * direct_jepa
            + progress * self.phase2_transport_loss_weight * transport_jepa
        )
        regularizers = self._phase2_transport_regularizers(transport_state)
        total_loss = self.phase1_token_loss_weight * token_jepa
        total_loss = total_loss + progress * (
            self.phase2_delay_prior_weight * regularizers["delay_prior_loss"]
            + self.phase2_monotonic_weight * regularizers["monotonic_loss"]
            + self.phase2_delay_smoothness_weight
            * regularizers["delay_smoothness_loss"]
            + self.phase2_match_mass_weight * regularizers["match_mass_loss"]
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
            "delay_prior": regularizers["delay_prior_loss"].item(),
            "monotonic": regularizers["monotonic_loss"].item(),
            "delay_smoothness": regularizers["delay_smoothness_loss"].item(),
            "match_mass_loss": regularizers["match_mass_loss"].item(),
            "delay_mean_ms": regularizers["delay_mean_ms"].item(),
            "delay_std_ms": regularizers["delay_std_ms"].item(),
            "transport_entropy": regularizers["transport_entropy"].item(),
            "matched_mass": regularizers["matched_mass"].item(),
            "minimum_matched_mass": regularizers[
                "minimum_matched_mass"
            ].item(),
            "unmatched_mass": regularizers["unmatched_mass"].item(),
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

    def compute_loss(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        ecg_stats: Optional[torch.Tensor] = None,
        collect_diagnostics: bool = False,
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
        if self.pretrain_phase == 2:
            return self._compute_phase2_loss(
                ecg, ppg, ecg_stats, collect_diagnostics
            )
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
