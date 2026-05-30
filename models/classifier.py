"""
Downstream classification head on top of pre-trained encoder.
"""
from typing import Optional, List

import torch
import torch.nn as nn

from .encoder import SignalEncoder


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
