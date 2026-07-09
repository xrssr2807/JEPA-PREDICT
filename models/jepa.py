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
        # ★ JETS 式掩码：预训练时随机丢弃信号片段
        mask_ratio: float = 0.0,         # 0=关闭, 0.7=保留30%
        mask_patch_size: int = 50,        # 每个patch的采样点数
        # New: auxiliary loss config
        use_stats_loss: bool = False,
        stats_loss_weight: float = 0.1,
        use_contrast_loss: bool = False,
        contrast_loss_weight: float = 1.0,
        use_token_align: bool = False,
        token_align_weight: float = 0.5,
        use_se: bool = False,
        use_inception: bool = False,
        vicreg_sim_weight: float = 1.0,
        vicreg_var_weight: float = 1.0,
        vicreg_cov_weight: float = 0.04,
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
            use_se=use_se,
            use_inception=use_inception,
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
        self.use_contrast_loss = use_contrast_loss
        self.contrast_loss_weight = contrast_loss_weight
        self.use_token_align = use_token_align
        self.token_align_weight = token_align_weight
        self.align_window = 3  # Soft-DTW 搜索窗口 (±3 token)
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

    def update_target_encoder(self, momentum: float):
        """EMA update target encoder towards context encoder."""
        ema_update(self.context_encoder, self.target_encoder, momentum)
        ema_update(self.context_proj, self.target_proj, momentum)
        # ★ M2AE 对比学习不需要EMA更新投影头（共享投影头无teacher）

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

    def _compute_contrast_loss(self, ecg_embed, ppg_embed):
        """M2AE InfoNCE: 对称对比ECG↔PPG"""
        if not hasattr(self, 'contrast_projector'):
            return torch.tensor(0.0, device=ecg_embed.device), {"contrast": 0.0}
        z_ecg = F.normalize(self.contrast_projector(ecg_embed), dim=-1)
        z_ppg = F.normalize(self.contrast_projector(ppg_embed), dim=-1)
        logits = torch.mm(z_ecg, z_ppg.t()) / self.contrast_temperature
        B = ecg_embed.size(0)
        labels = torch.arange(B, device=ecg_embed.device)
        l1 = F.cross_entropy(logits, labels)
        l2 = F.cross_entropy(logits.t(), labels)
        return (l1+l2)/2, {"contrast": (l1+l2).item()/2}

    def _compute_token_align_loss(self, ecg, ppg, token_mask=None,
                                   align_window=3, soft_temperature=0.1):
        """
        ★ Soft-DTW 风格弹性 Token 对齐 (替代原先的硬对齐).

        文档明确指出:
          "Soft-DTW 允许模型在一定的时间窗口内拉伸或压缩序列,
           寻找形态最相似的波峰进行匹配,完美吸收 PTT 波动"

        ECG 的 R 波和 PPG 的收缩峰之间有 ~200ms 生理延迟 (PTT),
        对应约 2 个 token (stride=16, 100Hz)。
        align_window=3 覆盖 ±300ms,足够容纳 PTT 变化。

        Args:
            token_mask: (B, 1, N) bool, True=可见位置
            align_window: 单侧搜索窗口 (±N token, 覆盖 PTT 生理范围)
            soft_temperature: softmax 温度, 越低对齐越"硬"
        """
        _, ecg_tokens = self.context_encoder(ecg, return_all=True)   # (B, N, D)
        _, ppg_tokens = self.target_encoder(ppg, return_all=True)    # (B, N, D)

        ppg_tokens = ppg_tokens.detach()  # teacher 停止梯度

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
            if token_mask is not None and not token_mask[0, 0, i]:
                continue

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
            total_loss += (1.0 - align_cos).sum()
            valid_count += 1

        loss = total_loss / max(valid_count * ecg_tokens.size(0), 1)
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

        # ★ JETS 式掩码：随机丢弃 ~70% patch，强制编码器从局部学习全局表征
        ecg_masked, token_mask = self._apply_jets_mask(ecg)

        # Get context embedding from MASKED ecg
        context_embed = self.forward_context(ecg_masked)  # (B, transformer_dim)

        # Get target embedding from FULL ppg (不作为掩码)
        target_embed = self.forward_target(ppg)  # (B, embedding_dim)

        # 1. JEPA prediction loss (ECG → PPG)
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

        # 3. ★ Soft-DTW 弹性 Token 对齐 (允许 ±3 token 的 PTT 生理延迟)
        if self.use_token_align:
            token_loss, token_info = self._compute_token_align_loss(
                ecg, ppg, token_mask=token_mask,
                align_window=self.align_window,
                soft_temperature=0.1,
            )
            total_loss = total_loss + self.token_align_weight * token_loss
            info.update(token_info)

        info["total_loss"] = total_loss.item()
        return total_loss, info
