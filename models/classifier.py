"""
Downstream classification head on top of pre-trained encoder.

Includes:
  - Vanilla MLP classifier (SignalClassifier, DualChannelClassifier)
  - CoT (Chain-of-Thought) reasoning head  ← from CWT-MAE v3
"""
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SignalEncoder


class DropPath(nn.Module):
    """Stochastic depth dropout."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class LatentReasoningHead(nn.Module):
    """
    Chain-of-Thought (CoT) classification head with learnable reasoning tokens.

    Adapted from CWT-MAE v3.

    Flow:
        Encoder features → [Reasoning Tokens] ─Cross-Attn→ Self-Attn → FFN
                                         ↓ Pool Query
                              → Decision Token → Linear → logits
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
        num_reasoning_tokens: int = 16,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens

        # Learnable reasoning tokens
        self.reasoning_tokens = nn.Parameter(
            torch.zeros(1, num_reasoning_tokens, embed_dim)
        )
        nn.init.normal_(self.reasoning_tokens, std=0.02)

        # Cross-attention: reasoning tokens ← encoder features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Self-attention: reasoning tokens interact
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Pooling query → decision token
        self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.pool_norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_encoder):
        """
        Args:
            x_encoder: (B, N, D) — encoder output tokens
        Returns:
            logits: (B, num_classes)
        """
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1)

        # 1. Cross-attention: reasoning tokens aggregate info from encoder
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + self.drop_path(attn_out))

        # 2. Self-attention: reasoning tokens reason with each other
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + self.drop_path(attn_out2))

        # 3. FFN
        queries = self.norm3(queries + self.drop_path(self.ffn(queries)))

        # 4. Pool to decision token
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, queries, queries)
        decision = self.pool_norm(pooled.squeeze(1))

        # 5. Classify
        return self.classifier(decision)


class SignalClassifier(nn.Module):
    """
    Pre-trained encoder + MLP classification head.

    Supports:
        - Linear probe: freeze encoder, train only classifier head
        - Full fine-tune: train everything
        - Progressive unfreeze: unfreeze layers gradually
    """

    def __init__(
        self,
        encoder: SignalEncoder,
        encoder_dim: int = 256,
        num_classes: int = 2,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.encoder = encoder

        # Classification head
        layers = []
        in_dim = encoder_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def freeze_encoder(self):
        """Freeze encoder for linear probe."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ):
        """
        Args:
            x: (B, 1, L)
            return_embedding: if True, also return encoder embedding
        Returns:
            logits: (B, num_classes)
            embedding: (B, encoder_dim) — only if return_embedding
        """
        embedding, _ = self.encoder(x)  # (B, encoder_dim)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits


class SignalClassifierCoT(nn.Module):
    """
    Single-channel classifier with CoT reasoning head.
    """
    def __init__(
        self,
        encoder: SignalEncoder,
        encoder_dim: int = 512,
        num_classes: int = 2,
        num_heads: int = 8,
        num_reasoning_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = LatentReasoningHead(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            num_reasoning_tokens=num_reasoning_tokens,
            dropout=dropout,
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        _, tokens = self.encoder(x, return_all=True)  # need per-token features for CoT
        tokens = F.layer_norm(tokens, tokens.shape[-1:])
        return self.head(tokens)


class DualChannelClassifierCoT(nn.Module):
    """
    Dual-channel classifier (ECG + PPG fusion) with CoT reasoning head.
    """
    def __init__(
        self,
        ecg_encoder: SignalEncoder,
        ppg_encoder: SignalEncoder,
        encoder_dim: int = 512,
        num_classes: int = 2,
        num_heads: int = 8,
        num_reasoning_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        # Concatenated tokens: (B, 2*N, D) → reasoning head
        self.head = LatentReasoningHead(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            num_reasoning_tokens=num_reasoning_tokens,
            dropout=dropout,
        )

    def freeze_encoders(self):
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False
        for p in self.ppg_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for p in self.ecg_encoder.parameters():
            p.requires_grad = True
        for p in self.ppg_encoder.parameters():
            p.requires_grad = True

    # 别名：兼容训练代码的 model.freeze_encoder() 调用
    freeze_encoder = freeze_encoders
    unfreeze_encoder = unfreeze_encoders

    def forward(self, ecg, ppg):
        _, ecg_tokens = self.ecg_encoder(ecg, return_all=True)
        _, ppg_tokens = self.ppg_encoder(ppg, return_all=True)
        ecg_tokens = F.layer_norm(ecg_tokens, ecg_tokens.shape[-1:])
        ppg_tokens = F.layer_norm(ppg_tokens, ppg_tokens.shape[-1:])
        all_tokens = torch.cat([ecg_tokens, ppg_tokens], dim=1)
        return self.head(all_tokens)


class DualChannelClassifier(nn.Module):
    """
    Classifier that fuses ECG and PPG encoders for downstream tasks.

    Both encoders are loaded from pre-trained JEPA weights.
    The fusion is simple: concat(ECG_embed, PPG_embed) → MLP → class.
    """

    def __init__(
        self,
        ecg_encoder: SignalEncoder,
        ppg_encoder: SignalEncoder,
        encoder_dim: int = 256,
        num_classes: int = 2,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        # Fusion classifier
        layers = []
        in_dim = encoder_dim * 2  # concat
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def freeze_encoders(self):
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False
        for param in self.ppg_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        for param in self.ecg_encoder.parameters():
            param.requires_grad = True
        for param in self.ppg_encoder.parameters():
            param.requires_grad = True

    def forward(
        self, ecg: torch.Tensor, ppg: torch.Tensor, return_embedding: bool = False
    ):
        ecg_embed, _ = self.ecg_encoder(ecg)
        ppg_embed, _ = self.ppg_encoder(ppg)
        fused = torch.cat([ecg_embed, ppg_embed], dim=-1)
        logits = self.classifier(fused)

        if return_embedding:
            return logits, fused
        return logits


class MultiScaleClassifier(nn.Module):
    """
    ★ HiMAE 风格多尺度分类头。

    HiMAE (Samsung) 论文核心发现：
    - 不同健康结局依赖不同时间尺度的PPG结构特征
    - PVC检测依赖粗尺度(L3), 实验室值偏好中尺度(L2)
    - 睡眠分期受益于局部+中尺度组合(L1+L2)

    三个尺度从编码器token序列中提取后拼接 → 分类。
    """

    def __init__(
        self,
        encoder: SignalEncoder,
        encoder_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder

        # 三尺度拼接 → MLP分类头
        total_dim = encoder_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: (B, 1, L) — PPG信号
        Returns:
            logits: (B, num_classes)
        """
        _, tokens = self.encoder(x, return_all=True)  # (B, N, D)
        B, N, D = tokens.shape

        # ── 尺度1 (细粒度): 每2个token池化 → 平均 ──
        # 捕获局部波形形态 (如PPG的切迹、上升支斜率)
        n1 = N // 2 * 2
        s1 = tokens[:, :n1].reshape(B, -1, 2, D).mean(dim=2).mean(dim=1)  # (B, D)

        # ── 尺度2 (中粒度): 每4个token池化 → 平均 ──
        # 捕获心跳级别的模式 (如心率变异性)
        n2 = N // 4 * 4
        s2 = tokens[:, :n2].reshape(B, -1, 4, D).mean(dim=2).mean(dim=1)  # (B, D)

        # ── 尺度3 (粗粒度): 全局平均 ──
        # 捕获整体趋势 (如基线漂移、长期变化)
        s3 = tokens.mean(dim=1)  # (B, D)

        # ── 拼接三个尺度 ──
        out = torch.cat([s1, s2, s3], dim=-1)  # (B, 3*D)
        logits = self.classifier(out)

        return logits


class SimpleFusion(nn.Module):
    """
    ★ M2AE 风格简单融合分类头 (替代 DualChannelClassifierCoT)。

    论文 "Biosignal Fingerprinting" (Oxford, 2026) 的核心设计:
    - 不用 token 拼接 (防止注意力稀释)
    - 每个编码器输出 pooled (B, 512) → concat → MLP

    ECG → Encoder → AvgPool → (B, 512) ─┐
                                          ├→ concat(1024) → BN → GELU → 256 → 分类
    PPG → Encoder → AvgPool → (B, 512) ─┘
    """

    def __init__(
        self,
        ecg_encoder: SignalEncoder,
        ppg_encoder: SignalEncoder,
        encoder_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),  # 1024→512
            nn.BatchNorm1d(encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim // 2),  # 512→256
            nn.BatchNorm1d(encoder_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_dim // 2, num_classes),
        )

    def freeze_encoder(self):
        """冻结两个编码器 (线性探测模式)."""
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False
        for p in self.ppg_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """解冻两个编码器 (全微调模式)."""
        for p in self.ecg_encoder.parameters():
            p.requires_grad = True
        for p in self.ppg_encoder.parameters():
            p.requires_grad = True

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        """
        Args:
            ecg: (B, 1, L) — ECG 信号
            ppg: (B, 1, L) — PPG 信号
        Returns:
            logits: (B, num_classes)
        """
        # 两个编码器各自输出 pooled embedding
        e, _ = self.ecg_encoder(ecg)  # (B, 512)
        p, _ = self.ppg_encoder(ppg)  # (B, 512)

        # 拼接 + 分类
        fused = torch.cat([e, p], dim=-1)  # (B, 1024)
        return self.fusion(fused)
