import argparse
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from models.classifier import DiseaseConditionedMILHead
from models.losses import AsymmetricLoss
from official_fm_baselines.common import (
    DISEASE_LABELS,
    EmbeddingCache,
    PatientEmbeddingDataset,
    ensure_patient_counts,
    seed_everything,
)


class CachedEmbeddingMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mil = DiseaseConditionedMILHead(hidden_dim, num_classes, dropout)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor):
        segment_repr = self.projector(bag)
        logits, _, _ = self.mil(segment_repr, mask)
        return logits


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray):
    aucs = []
    aps = []
    for index in range(labels.shape[1]):
        target = labels[:, index]
        if np.unique(target).size < 2:
            aucs.append(float("nan"))
            aps.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(target, probabilities[:, index])))
            aps.append(float(average_precision_score(target, probabilities[:, index])))
    return {
        "macro_auc": float(np.nanmean(aucs)),
        "macro_auprc": float(np.nanmean(aps)),
        "chd_auc": aucs[4],
        "chd_auprc": aps[4],
        "auc_per_class": aucs,
        "auprc_per_class": aps,
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    labels = []
    probabilities = []
    uids = []
    for bag, target, uid, mask in loader:
        bag = bag.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(bag, mask)
        loss = criterion(logits, target)
        losses.append(float(loss.item()) * target.shape[0])
        labels.append(target.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        uids.extend([str(value) for value in uid])
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    metrics = compute_metrics(labels, probabilities)
    metrics["loss"] = sum(losses) / len(uids)
    return metrics, labels, probabilities, uids


def focus_auc_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.2):
    scores = logits[:, 4]
    target = labels[:, 4] > 0.5
    positive = scores[target]
    negative = scores[~target]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    differences = positive[:, None] - negative[None, :]
    return torch.relu(margin - differences).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_segments", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = EmbeddingCache.load(os.path.join(args.cache_dir, "train_embeddings.pt"))
    val_cache = EmbeddingCache.load(os.path.join(args.cache_dir, "val_embeddings.pt"))
    ensure_patient_counts(train_cache, 5387, "train")
    ensure_patient_counts(val_cache, 1155, "val")
    if train_cache.embeddings.shape[-1] != val_cache.embeddings.shape[-1]:
        raise RuntimeError("Train/validation embedding dimensions differ")

    train_dataset = PatientEmbeddingDataset(train_cache, args.max_segments, train=True)
    val_dataset = PatientEmbeddingDataset(val_cache, args.max_segments, train=False)
    common_loader = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        common_loader["prefetch_factor"] = 4
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **common_loader,
    )

    model = CachedEmbeddingMIL(
        input_dim=int(train_cache.embeddings.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_classes=len(DISEASE_LABELS),
        dropout=args.dropout,
    ).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    best_state = None
    best_score = -math.inf
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for bag, target, _, mask in train_loader:
            bag = bag.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(bag, mask)
                loss = criterion(logits, target)
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                    logits[:, 4], target[:, 4]
                )
                loss = loss + 0.1 * focus_auc_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()

        metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        score = 0.7 * metrics["chd_auc"] + 0.3 * metrics["macro_auc"]
        record = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "selection_score": score,
            **metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d} train={record['train_loss']:.5f} "
            f"val={metrics['loss']:.5f} macro_auc={metrics['macro_auc']:.4f} "
            f"chd_auc={metrics['chd_auc']:.4f} chd_auprc={metrics['chd_auprc']:.4f}"
        )
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[EarlyStop] no improvement for {stale} epochs")
                break

    model.load_state_dict(best_state)
    metrics, labels, probabilities, uids = evaluate(model, val_loader, criterion, device)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": int(train_cache.embeddings.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "labels": DISEASE_LABELS,
            "seed": args.seed,
            "validation_metrics": metrics,
            "test_set_used": False,
        },
        os.path.join(args.output_dir, "best_validation_model.pt"),
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "validation_patient_predictions.npz"),
        uid=np.asarray(uids),
        labels=labels,
        probabilities=probabilities,
    )
    summary = {
        "seed": args.seed,
        "model_selection": "0.7*CHD_AUC + 0.3*Macro_AUC",
        "validation_metrics": metrics,
        "history": history,
        "test_set_used": False,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "DEVELOPMENT_COMPLETE"), "w", encoding="ascii") as handle:
        handle.write("test_set_used=false\n")
    print(json.dumps(summary["validation_metrics"], ensure_ascii=False, indent=2))
    print(f"[Complete] validation-only cached MIL: {args.output_dir}")


if __name__ == "__main__":
    main()

