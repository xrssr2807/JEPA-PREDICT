"""
Loss functions for downstream fine-tuning.

Includes:
  - FocalLoss (binary & multi-class)
  - MultiLabelFocalLoss
  - AsymmetricLoss (from ASL paper, for multi-label)

Adapted from CWT-MAE v3 (Wearable-Foundation-Model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (0 = standard CE loss)
        alpha: class weights (tensor of shape (num_classes,))
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: float in [0, 1]
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C)
            targets: (B,) long tensor of class indices
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    FL = -(1 - p_t)^gamma * log(p_t)
    where p_t = sigmoid(logit) if y=1 else 1-sigmoid(logit)
    """
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits
            targets: (B, C) float (multi-hot) or (B,) long
        """
        if targets.dtype == torch.long:
            targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        else:
            targets_onehot = targets

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_onehot, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = targets_onehot * probs + (1 - targets_onehot) * (1 - probs)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = targets_onehot * self.alpha + (1 - targets_onehot) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    From "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)

    Key: gamma_neg > gamma_pos → easy negatives down-weighted more aggressively.
         + probability shift (margin m) eliminates contribution of very-low-prob negatives.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=False, pos_weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C)
            targets: (B, C) float multi-hot
        """
        p = torch.sigmoid(logits)

        # Positive loss
        los_pos = targets * F.logsigmoid(logits)
        if self.gamma_pos > 0:
            pt_pos = (1 - p)
            if self.disable_torch_grad_focal_loss:
                pt_pos = pt_pos.detach()
            los_pos = los_pos * (pt_pos ** self.gamma_pos)

        # Negative loss with probability shifting
        p_m = torch.clamp(p - self.clip, min=0.0)
        los_neg = (1 - targets) * torch.log(torch.clamp(1 - p_m, min=self.eps))
        if self.gamma_neg > 0:
            pt_neg = p_m
            if self.disable_torch_grad_focal_loss:
                pt_neg = pt_neg.detach()
            los_neg = los_neg * (pt_neg ** self.gamma_neg)

        loss = los_pos + los_neg

        if self.pos_weight is not None:
            loss = loss * self.pos_weight

        return -loss.sum(dim=-1).mean()


def compute_pos_weight(train_dataset, num_classes: int, device: str = "cpu") -> torch.Tensor:
    """
    Compute per-class positive weights for imbalanced classification.

    pos_weight[c] = sqrt(N_neg[c] / N_pos[c]), clipped to [1.0, 50.0].

    Uses sqrt to smooth extreme ratios.
    """
    import numpy as np
    from torch.utils.data import DataLoader

    # Count labels in training set
    label_counts = np.zeros(num_classes, dtype=np.float64)
    total = 0

    print(f"[PosWeight] Scanning {len(train_dataset)} samples for class distribution...")
    for i in range(len(train_dataset)):
        try:
            _, label = train_dataset[i]
            if isinstance(label, int):
                label_counts[label] += 1
            else:
                label_counts[label] += 1  # Already one-hot or multi-label
        except Exception:
            pass
        total += 1
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(train_dataset)}")

    # For binary (single-label), per-class counts
    if num_classes == 2 and label_counts.sum() > 0:
        pos_counts = label_counts
        neg_counts = total - pos_counts
        pos_counts = np.maximum(pos_counts, 1)
        # sqrt smoothing
        weights = np.sqrt(neg_counts / pos_counts)
        weights = np.clip(weights, 1.0, 50.0)
    else:
        # Multi-label: each class independently
        pos_counts = label_counts
        neg_counts = total - label_counts
        pos_counts = np.maximum(pos_counts, 1)
        weights = np.sqrt(neg_counts / pos_counts)
        weights = np.clip(weights, 1.0, 50.0)

    print(f"[PosWeight] Distribution: {dict(zip(range(num_classes), label_counts.astype(int)))}")
    print(f"[PosWeight] pos_weight: {[round(w, 2) for w in weights.tolist()]}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(loss_type: str, num_classes: int, pos_weight=None,
                    gamma: float = 2.0, gamma_neg: int = 4, gamma_pos: int = 1,
                    clip: float = 0.05, label_smoothing: float = 0.0):
    """
    Build loss criterion from config parameters.

    Args:
        loss_type: "ce" | "focal" | "asl" | "bce"
        num_classes: number of output classes
        pos_weight: (num_classes,) tensor or None
        gamma: focal gamma
        gamma_neg, gamma_pos: ASL parameters
        clip: ASL probability shift
        label_smoothing: label smoothing for CE/FocalLoss
    """
    if loss_type == "asl":
        return AsymmetricLoss(
            gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
            pos_weight=pos_weight
        )
    elif loss_type == "focal":
        return FocalLoss(
            gamma=gamma, alpha=None, reduction='mean',
            label_smoothing=label_smoothing,
        )
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:  # default: cross entropy
        if pos_weight is not None:
            return nn.CrossEntropyLoss(weight=pos_weight, label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
