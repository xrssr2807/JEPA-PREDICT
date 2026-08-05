"""Capacity-controlled ECG/PPG masked reconstruction baselines.

The encoders are the same ``SignalEncoder`` instances used by JEPA.  Only the
pre-training objective changes:

* ``multimodal_mae`` masks both modalities and reconstructs both directions.
* ``xmae_objective`` keeps PPG visible, continuously masks ECG, and uses
  directional PPG-to-ECG cross-attention to reconstruct masked ECG patches.

The second mode is deliberately named an objective-level reproduction: it
preserves the central xMAE training constraint while controlling encoder
capacity for a fair comparison with PhysioV2.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_patch_mask(
    batch_size: int,
    num_patches: int,
    ratio: float,
    mode: str,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return a boolean patch mask where ``True`` means hidden."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"mask ratio must be in [0, 1], got {ratio}")
    if num_patches < 1:
        raise ValueError("num_patches must be positive")
    hidden = min(num_patches, max(0, round(ratio * num_patches)))
    if hidden == 0:
        return torch.zeros(
            batch_size, num_patches, dtype=torch.bool, device=device
        )
    if hidden == num_patches:
        return torch.ones(
            batch_size, num_patches, dtype=torch.bool, device=device
        )

    if mode == "scatter":
        scores = torch.rand(
            batch_size,
            num_patches,
            device=device,
            generator=generator,
        )
        indices = scores.argsort(dim=1)[:, :hidden]
        mask = torch.zeros(
            batch_size, num_patches, dtype=torch.bool, device=device
        )
        return mask.scatter(1, indices, True)

    positions = torch.arange(num_patches, device=device).view(1, -1)
    if mode == "block":
        starts = torch.randint(
            0,
            num_patches - hidden + 1,
            (batch_size, 1),
            device=device,
            generator=generator,
        )
        return (positions >= starts) & (positions < starts + hidden)
    if mode == "anchor":
        visible = num_patches - hidden
        starts = torch.randint(
            0,
            num_patches - visible + 1,
            (batch_size, 1),
            device=device,
            generator=generator,
        )
        return (positions < starts) | (positions >= starts + visible)
    raise ValueError(f"Unknown mask mode: {mode}")


def patchify_waveform(
    waveform: torch.Tensor, num_patches: int
) -> torch.Tensor:
    """Convert ``(B, 1, L)`` waveforms to ``(B, N, ceil(L/N))`` patches."""
    if waveform.dim() != 3 or waveform.size(1) != 1:
        raise ValueError(
            f"Expected waveform shape (B, 1, L), got {tuple(waveform.shape)}"
        )
    patch_size = (waveform.size(-1) + num_patches - 1) // num_patches
    padded_length = patch_size * num_patches
    waveform = F.pad(waveform, (0, padded_length - waveform.size(-1)))
    return waveform[:, 0].reshape(waveform.size(0), num_patches, patch_size)


def apply_patch_mask_to_waveform(
    waveform: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Hide raw samples before the CNN to prevent trivial local leakage."""
    patch_size = (waveform.size(-1) + mask.size(1) - 1) // mask.size(1)
    sample_mask = mask.repeat_interleave(patch_size, dim=1)
    sample_mask = sample_mask[:, : waveform.size(-1)].unsqueeze(1)
    return waveform.masked_fill(sample_mask, 0.0)


def masked_patch_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean reconstruction error over masked patches only."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    if mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"mask {tuple(mask.shape)} != tokens {tuple(prediction.shape[:2])}"
        )
    per_patch = (prediction - target).square().mean(dim=-1)
    weights = mask.to(dtype=per_patch.dtype)
    return (per_patch * weights).sum() / weights.sum().clamp_min(1.0)


class CrossAttentionDecoderBlock(nn.Module):
    """Pre-norm cross-attention decoder block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        update = self.cross_attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )[0]
        query = query + self.dropout(update)
        return query + self.ff(self.ff_norm(query))


class CrossModalDecoder(nn.Module):
    """Cross-modal decoder followed by per-token waveform prediction."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        depth: int,
        max_patch_size: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionDecoderBlock(dim, heads, ff_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.waveform_head = nn.Linear(dim, max_patch_size)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            query = block(query, context)
        output = self.waveform_head(self.norm(query))
        if patch_size > output.size(-1):
            raise ValueError(
                f"patch_size={patch_size} exceeds decoder capacity "
                f"{output.size(-1)}"
            )
        return output[..., :patch_size]


class CrossModalMaskedAutoencoder(nn.Module):
    """Shared-backbone multimodal MAE and xMAE-objective baseline."""

    VALID_OBJECTIVES = {"multimodal_mae", "xmae_objective"}

    def __init__(
        self,
        ecg_encoder: nn.Module,
        ppg_encoder: nn.Module,
        objective: str,
        model_dim: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        decoder_depth: int = 2,
        max_patch_size: int = 32,
    ):
        super().__init__()
        if objective not in self.VALID_OBJECTIVES:
            raise ValueError(f"Unknown reconstruction objective: {objective}")
        self.objective = objective
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        self.ecg_mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.ppg_mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.ecg_mask_token, std=0.02)
        nn.init.trunc_normal_(self.ppg_mask_token, std=0.02)
        self.ppg_to_ecg = CrossModalDecoder(
            model_dim,
            heads,
            ff_dim,
            dropout,
            decoder_depth,
            max_patch_size,
        )
        self.ecg_to_ppg = (
            CrossModalDecoder(
                model_dim,
                heads,
                ff_dim,
                dropout,
                decoder_depth,
                max_patch_size,
            )
            if objective == "multimodal_mae"
            else None
        )

    def token_count(self, waveform_length: int) -> int:
        """Infer the CNN token count without an extra encoder forward pass."""
        length = int(waveform_length)
        for block in self.ecg_encoder.cnn.conv_blocks:
            convolution = block[0]
            kernel = convolution.kernel_size[0]
            stride = convolution.stride[0]
            padding = convolution.padding[0]
            dilation = convolution.dilation[0]
            length = (
                length + 2 * padding - dilation * (kernel - 1) - 1
            ) // stride + 1
        return length

    def forward(
        self,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        ecg_mask: torch.Tensor,
        ppg_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ecg_target_waveform = ecg
        ppg_target_waveform = ppg
        ecg = apply_patch_mask_to_waveform(ecg, ecg_mask)
        ppg = apply_patch_mask_to_waveform(ppg, ppg_mask)
        ecg_input_tokens = self.ecg_encoder.tokenize(ecg)
        ppg_input_tokens = self.ppg_encoder.tokenize(ppg)
        if ecg_input_tokens.shape[:2] != ppg_input_tokens.shape[:2]:
            raise ValueError("ECG and PPG token grids must match")
        if ecg_mask.shape != ecg_input_tokens.shape[:2]:
            raise ValueError("ECG mask does not match encoder token grid")
        if ppg_mask.shape != ppg_input_tokens.shape[:2]:
            raise ValueError("PPG mask does not match encoder token grid")

        _, ecg_tokens = self.ecg_encoder.encode_tokens(
            ecg_input_tokens,
            return_all=True,
            token_mask=ecg_mask,
            mask_token=self.ecg_mask_token,
        )
        _, ppg_tokens = self.ppg_encoder.encode_tokens(
            ppg_input_tokens,
            return_all=True,
            token_mask=ppg_mask,
            mask_token=self.ppg_mask_token,
        )
        ecg_target = patchify_waveform(
            ecg_target_waveform, ecg_tokens.size(1)
        )
        ppg_target = patchify_waveform(
            ppg_target_waveform, ppg_tokens.size(1)
        )
        ecg_prediction = self.ppg_to_ecg(
            ecg_tokens, ppg_tokens, ecg_target.size(-1)
        )
        output = {
            "ecg_prediction": ecg_prediction,
            "ecg_target": ecg_target,
            "ecg_mask": ecg_mask,
        }
        if self.ecg_to_ppg is not None:
            output.update({
                "ppg_prediction": self.ecg_to_ppg(
                    ppg_tokens, ecg_tokens, ppg_target.size(-1)
                ),
                "ppg_target": ppg_target,
                "ppg_mask": ppg_mask,
            })
        return output

    def compute_loss(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        ecg_loss = masked_patch_mse(
            output["ecg_prediction"],
            output["ecg_target"],
            output["ecg_mask"],
        )
        if self.objective == "xmae_objective":
            ppg_loss = ecg_loss.new_zeros(())
            total = ecg_loss
        else:
            ppg_loss = masked_patch_mse(
                output["ppg_prediction"],
                output["ppg_target"],
                output["ppg_mask"],
            )
            total = 0.5 * (ecg_loss + ppg_loss)
        return {"total": total, "ecg": ecg_loss, "ppg": ppg_loss}
