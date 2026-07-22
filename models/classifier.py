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


class DualChannelSimpleFusion(nn.Module):
    """
    ★ SimpleFusion: ECG+PPG 向量级融合 (Biosignal Fingerprinting 风格)

    ECG → Encoder → avg_pool → (B, 512)
    PPG → Encoder → avg_pool → (B, 512)
    concat → Linear(512)→GELU→Linear(256)→GELU→Linear(2)

    优势: 避免CoT的124-token注意力稀释, 参数少收敛快
    """
    def __init__(
        self,
        ecg_encoder: SignalEncoder,
        ppg_encoder: SignalEncoder,
        encoder_dim: int = 512,
        num_classes: int = 2,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        # Fusion MLP
        layers = []
        in_dim = encoder_dim * 2  # 512 + 512 = 1024
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.fusion = nn.Sequential(*layers)

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

    freeze_encoder = freeze_encoders
    unfreeze_encoder = unfreeze_encoders

    def unfreeze_ppg_only(self):
        """★ CardioPPG风格: 只解冻PPG编码器, ECG保持冻结 (防止过拟合)"""
        for p in self.ppg_encoder.parameters():
            p.requires_grad = True
        # ECG encoder stays frozen

    def forward(self, ecg, ppg):
        e, _ = self.ecg_encoder(ecg)  # (B, 512)
        p, _ = self.ppg_encoder(ppg)  # (B, 512)
        fused = torch.cat([e, p], dim=-1)  # (B, 1024)
        return self.fusion(fused)


class AsymmetricFusion(nn.Module):
    """
    ★ 不对称双通道融合: ECG冻结(辅助) + PPG LoRA微调(主力)。

    ECG → context_encoder(frozen) → (B, 512) ─┐
                                                 ├─ concat → MLP → 2
    PPG → target_encoder(LoRA)    → (B, 512) ─┘

    仅 ~1M 可训参数, 防过拟合, ECG仅作为辅助信号。
    """
    def __init__(
        self,
        ecg_encoder, ppg_encoder,
        encoder_dim: int = 512,
        num_classes: int = 2,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        # ECG encoder 始终冻结
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False

        # Fusion MLP
        layers = []
        in_dim = encoder_dim * 2  # 512 + 512 = 1024
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.fusion = nn.Sequential(*layers)

    def forward(self, ecg, ppg):
        with torch.no_grad():
            e, _ = self.ecg_encoder(ecg)  # (B, 512), ECG冻结
        p, _ = self.ppg_encoder(ppg)       # (B, 512), PPG LoRA微调
        fused = torch.cat([e, p], dim=-1)
        return self.fusion(fused)

    def freeze_encoder(self):
        """冻结PPG encoder的非LoRA参数 (Probe阶段用)"""
        for p in self.ppg_encoder.parameters():
            p.requires_grad = False

    def unfreeze_ppg_lora(self):
        """只解冻PPG encoder的LoRA参数"""
        for n, p in self.ppg_encoder.named_parameters():
            if 'lora' in n:
                p.requires_grad = True

    freeze_encoder_alias = freeze_encoder
    unfreeze_encoder = freeze_encoder  # 默认不解冻


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


class TemporalPyramidPool(nn.Module):
    """Learned pooling over progressively smoothed temporal token sequences."""

    def __init__(self, dim: int, scales=(1, 2, 4)):
        super().__init__()
        self.scales = tuple(scales)
        self.queries = nn.Parameter(torch.empty(len(self.scales), dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.scales])
        nn.init.normal_(self.queries, std=dim ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"TemporalPyramidPool expects (B,N,D), got {tuple(tokens.shape)}")
        x = tokens.transpose(1, 2)
        outputs = []
        score_scale = tokens.size(-1) ** -0.5
        for idx, kernel in enumerate(self.scales):
            if kernel > 1 and tokens.size(1) >= kernel:
                pooled = F.avg_pool1d(
                    x, kernel_size=kernel, stride=max(1, kernel // 2), ceil_mode=True,
                ).transpose(1, 2)
            else:
                pooled = tokens
            pooled = self.norms[idx](pooled)
            scores = torch.einsum("bnd,d->bn", pooled, self.queries[idx]) * score_scale
            weights = torch.softmax(scores, dim=1)
            outputs.append(torch.sum(pooled * weights.unsqueeze(-1), dim=1))
        return torch.cat(outputs, dim=-1)


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
        self.temporal_pyramid = TemporalPyramidPool(encoder_dim)

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
        out = self.temporal_pyramid(tokens)
        logits = self.classifier(out)

        return logits


class DiseaseConditionedMILHead(nn.Module):
    """One segment-attention distribution and decision head per disease."""

    def __init__(self, dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, num_classes),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.weight)

    @staticmethod
    def _masked_attention(scores: torch.Tensor, segment_mask: torch.Tensor = None):
        if segment_mask is not None:
            if segment_mask.shape != scores.shape[:2]:
                raise ValueError(
                    f"segment_mask must have shape {tuple(scores.shape[:2])}, "
                    f"got {tuple(segment_mask.shape)}"
                )
            mask = segment_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(-1)
            if not torch.all(mask.any(dim=1)):
                raise ValueError("Every patient bag must contain at least one valid segment")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1)

    def forward(self, segment_repr: torch.Tensor, segment_mask: torch.Tensor = None):
        attention = self._masked_attention(self.attention(segment_repr), segment_mask)
        patient_repr = torch.einsum("bsk,bsd->bkd", attention, segment_repr)
        patient_repr = self.dropout(self.norm(patient_repr))
        logits = torch.einsum("bkd,kd->bk", patient_repr, self.weight) + self.bias
        return logits, patient_repr, attention


class DiseaseConditionedModalityMILHead(nn.Module):
    """Per-disease temporal attention followed by per-disease ECG/PPG fusion."""

    def __init__(self, dim: int, num_classes: int, dropout: float):
        super().__init__()
        hidden = max(dim // 2, 1)
        self.ecg_attention = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, num_classes),
        )
        self.ppg_attention = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, num_classes),
        )
        self.disease_embedding = nn.Parameter(torch.empty(num_classes, dim))
        self.modality_gate = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.interaction = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.disease_embedding, std=0.02)
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        ecg_segment_repr: torch.Tensor,
        ppg_segment_repr: torch.Tensor,
        segment_mask: torch.Tensor = None,
    ):
        ecg_attention = DiseaseConditionedMILHead._masked_attention(
            self.ecg_attention(ecg_segment_repr), segment_mask,
        )
        ppg_attention = DiseaseConditionedMILHead._masked_attention(
            self.ppg_attention(ppg_segment_repr), segment_mask,
        )
        ecg_patient = torch.einsum("bsk,bsd->bkd", ecg_attention, ecg_segment_repr)
        ppg_patient = torch.einsum("bsk,bsd->bkd", ppg_attention, ppg_segment_repr)

        disease = self.disease_embedding.unsqueeze(0).expand(ecg_patient.size(0), -1, -1)
        gate = torch.sigmoid(self.modality_gate(torch.cat([
            ecg_patient, ppg_patient, disease,
        ], dim=-1)))
        cross = self.interaction(torch.cat([
            torch.abs(ecg_patient - ppg_patient), ecg_patient * ppg_patient,
        ], dim=-1))
        patient_repr = self.dropout(self.norm(
            gate * ecg_patient + (1.0 - gate) * ppg_patient + cross
        ))
        logits = torch.einsum("bkd,kd->bk", patient_repr, self.weight) + self.bias
        attention = 0.5 * (ecg_attention + ppg_attention)
        return logits, patient_repr, attention, gate


class PatientMILClassifier(nn.Module):
    """
    Patient-level multi-instance classifier.

    Input is a bag of segments per patient: (B, S, C, L). Each segment is
    encoded independently, then an attention pooling head aggregates segment
    embeddings into one patient representation.
    """

    def __init__(
        self,
        encoder: SignalEncoder,
        encoder_dim: int = 512,
        num_classes: int = 9,
        use_multiscale: bool = True,
        dropout: float = 0.3,
        encoder_chunk_size: int = 0,
        input_channel: int = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_multiscale = use_multiscale
        self.encoder_chunk_size = int(encoder_chunk_size or 0)
        self.input_channel = None if input_channel is None else int(input_channel)

        rep_dim = encoder_dim * 3 if use_multiscale else encoder_dim
        self.temporal_pyramid = TemporalPyramidPool(encoder_dim) if use_multiscale else None
        self.segment_proj = nn.Sequential(
            nn.Linear(rep_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mil_head = DiseaseConditionedMILHead(encoder_dim, num_classes, dropout)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        chunk_size = self.encoder_chunk_size
        if chunk_size <= 0 or flat.size(0) <= chunk_size:
            if self.use_multiscale:
                _, tokens = self.encoder(flat, return_all=True)
                return self.temporal_pyramid(tokens)
            segment_repr, _ = self.encoder(flat)
            return segment_repr

        reps = []
        for start in range(0, flat.size(0), chunk_size):
            chunk = flat[start:start + chunk_size]
            if self.use_multiscale:
                _, tokens = self.encoder(chunk, return_all=True)
                reps.append(self.temporal_pyramid(tokens))
            else:
                segment_repr, _ = self.encoder(chunk)
                reps.append(segment_repr)
        return torch.cat(reps, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        segment_mask: torch.Tensor = None,
    ):
        if x.dim() != 4:
            raise ValueError(f"PatientMILClassifier expects (B,S,C,L), got {tuple(x.shape)}")
        B, S, C, L = x.shape
        if self.input_channel is not None:
            if not 0 <= self.input_channel < C:
                raise ValueError(
                    f"input_channel={self.input_channel} is invalid for {C} input channels"
                )
            x = x[:, :, self.input_channel:self.input_channel + 1]
            C = 1
        flat = x.reshape(B * S, C, L)
        segment_repr = self._encode_flat(flat)

        segment_repr = self.segment_proj(segment_repr).reshape(B, S, -1)
        logits, patient_repr, _ = self.mil_head(segment_repr, segment_mask=segment_mask)

        if return_embedding:
            return logits, patient_repr.mean(dim=1)
        return logits


class DualStreamPatientMILClassifier(nn.Module):
    """Patient-level fusion using ECG-context and PPG-target JEPA encoders."""

    def __init__(self, ecg_encoder: SignalEncoder, ppg_encoder: SignalEncoder,
                 encoder_dim: int = 512, num_classes: int = 9,
                 use_multiscale: bool = True, dropout: float = 0.3,
                 encoder_chunk_size: int = 0, ppg_channel: int = 0,
                 ecg_channel: int = 1,
                 disease_conditioned_fusion: bool = True):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        self.use_multiscale = use_multiscale
        self.encoder_chunk_size = int(encoder_chunk_size or 0)
        self.ppg_channel = int(ppg_channel)
        self.ecg_channel = int(ecg_channel)
        self.disease_conditioned_fusion = bool(disease_conditioned_fusion)

        rep_dim = encoder_dim * 3 if use_multiscale else encoder_dim
        self.ecg_pyramid = TemporalPyramidPool(encoder_dim) if use_multiscale else None
        self.ppg_pyramid = TemporalPyramidPool(encoder_dim) if use_multiscale else None
        self.ecg_proj = nn.Sequential(nn.Linear(rep_dim, encoder_dim), nn.LayerNorm(encoder_dim))
        self.ppg_proj = nn.Sequential(nn.Linear(rep_dim, encoder_dim), nn.LayerNorm(encoder_dim))
        if self.disease_conditioned_fusion:
            self.modality_mil_head = DiseaseConditionedModalityMILHead(
                encoder_dim, num_classes, dropout,
            )
        else:
            # Legacy branch retained so historical downstream teachers remain loadable.
            self.modality_gate = nn.Linear(encoder_dim * 2, encoder_dim)
            self.interaction = nn.Sequential(
                nn.Linear(encoder_dim * 2, encoder_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.segment_norm = nn.LayerNorm(encoder_dim)
            self.mil_head = DiseaseConditionedMILHead(encoder_dim, num_classes, dropout)

    def freeze_encoder(self):
        for encoder in (self.ecg_encoder, self.ppg_encoder):
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for encoder in (self.ecg_encoder, self.ppg_encoder):
            for param in encoder.parameters():
                param.requires_grad = True

    def shared_encoder_parameters(self):
        """Yield each shared online encoder parameter exactly once."""
        seen = set()
        for encoder in (self.ecg_encoder, self.ppg_encoder):
            for param in encoder.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    yield param

    def head_parameters(self):
        """Yield task-head parameters without the shared JEPA encoders."""
        encoder_ids = {id(param) for param in self.shared_encoder_parameters()}
        for param in self.parameters():
            if id(param) not in encoder_ids:
                yield param

    def _encode_one(self, encoder, pyramid, signal):
        if self.use_multiscale:
            _, tokens = encoder(signal, return_all=True)
            return pyramid(tokens)
        pooled, _ = encoder(signal)
        return pooled

    def _encode_flat(self, flat: torch.Tensor):
        chunk_size = self.encoder_chunk_size or flat.size(0)
        ecg_reps, ppg_reps = [], []
        for start in range(0, flat.size(0), chunk_size):
            chunk = flat[start:start + chunk_size]
            ppg = chunk[:, self.ppg_channel:self.ppg_channel + 1]
            ecg = chunk[:, self.ecg_channel:self.ecg_channel + 1]
            ecg_reps.append(self._encode_one(self.ecg_encoder, self.ecg_pyramid, ecg))
            ppg_reps.append(self._encode_one(self.ppg_encoder, self.ppg_pyramid, ppg))
        return torch.cat(ecg_reps, dim=0), torch.cat(ppg_reps, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        segment_mask: torch.Tensor = None,
    ):
        if x.dim() != 4:
            raise ValueError(
                f"DualStreamPatientMILClassifier expects (B,S,C,L), got {tuple(x.shape)}"
            )
        B, S, C, L = x.shape
        required_channels = max(self.ppg_channel, self.ecg_channel) + 1
        if C < required_channels:
            raise ValueError(f"Expected at least {required_channels} channels, got {C}")

        ecg_repr, ppg_repr = self._encode_flat(x.reshape(B * S, C, L))
        ecg_repr = self.ecg_proj(ecg_repr).reshape(B, S, -1)
        ppg_repr = self.ppg_proj(ppg_repr).reshape(B, S, -1)
        if self.disease_conditioned_fusion:
            logits, patient_repr, _, _ = self.modality_mil_head(
                ecg_repr, ppg_repr, segment_mask=segment_mask,
            )
        else:
            flat_ecg = ecg_repr.reshape(B * S, -1)
            flat_ppg = ppg_repr.reshape(B * S, -1)
            gate = torch.sigmoid(self.modality_gate(torch.cat([flat_ecg, flat_ppg], dim=-1)))
            cross = self.interaction(torch.cat([
                torch.abs(flat_ecg - flat_ppg), flat_ecg * flat_ppg,
            ], dim=-1))
            segment_repr = self.segment_norm(
                gate * flat_ecg + (1.0 - gate) * flat_ppg + cross
            ).reshape(B, S, -1)
            logits, patient_repr, _ = self.mil_head(
                segment_repr, segment_mask=segment_mask,
            )
        if return_embedding:
            return logits, patient_repr.mean(dim=1)
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
