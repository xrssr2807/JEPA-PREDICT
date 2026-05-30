"""
Downstream fine-tuning: linear probe → full fine-tune on CHD classification.
Supports single-channel (ECG or PPG) and dual-channel (ECG + PPG) classification.
"""
import os
import time
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset.data import DownstreamDataset
from models.encoder import SignalEncoder
from models.classifier import SignalClassifier, DualChannelClassifier


def build_downstream_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    modality: str,  # "ecg", "ppg", or "dual"
) -> tuple:
    """Build train and test dataloaders for downstream fine-tuning."""

    if modality == "ecg":
        data_dir = data_config.chd_ecg_dir + "/ecg_chd"
        split_file = data_config.chd_ecg_dir + "/train_test_split.json"
    elif modality == "ppg":
        data_dir = data_config.chd_ppg_dir + "/ppg_chd"
        split_file = data_config.chd_ppg_dir + "/train_test_split.json"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    train_dataset = DownstreamDataset(
        data_dir=data_dir,
        split_file=split_file,
        split="train",
        normalize=data_config.normalize,
    )
    test_dataset = DownstreamDataset(
        data_dir=data_dir,
        split_file=split_file,
        split="test",
        normalize=data_config.normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.downstream_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.downstream_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def build_dual_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
):
    """Build paired ECG+PPG dataloaders for dual-channel classification."""
    # ECG
    ecg_dir = data_config.chd_ecg_dir + "/ecg_chd"
    ecg_split = data_config.chd_ecg_dir + "/train_test_split.json"
    # PPG
    ppg_dir = data_config.chd_ppg_dir + "/ppg_chd"
    ppg_split = data_config.chd_ppg_dir + "/train_test_split.json"

    # We need a paired dataset wrapper
    from torch.utils.data import Dataset as TorchDataset

    class PairedDataset(TorchDataset):
        def __init__(self, ecg_ds, ppg_ds):
            self.ecg_ds = ecg_ds
            self.ppg_ds = ppg_ds
            assert len(ecg_ds) == len(ppg_ds), "Datasets must have same length"

        def __len__(self):
            return len(self.ecg_ds)

        def __getitem__(self, idx):
            ecg_data, ecg_label = self.ecg_ds[idx]
            ppg_data, ppg_label = self.ppg_ds[idx]
            # Labels should match (same patient, same segment)
            return ecg_data, ppg_data, ecg_label

    ecg_train = DownstreamDataset(ecg_dir, ecg_split, "train", data_config.normalize)
    ppg_train = DownstreamDataset(ppg_dir, ppg_split, "train", data_config.normalize)
    ecg_test = DownstreamDataset(ecg_dir, ecg_split, "test", data_config.normalize)
    ppg_test = DownstreamDataset(ppg_dir, ppg_split, "test", data_config.normalize)

    train_loader = DataLoader(
        PairedDataset(ecg_train, ppg_train),
        batch_size=train_config.downstream_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        PairedDataset(ecg_test, ppg_test),
        batch_size=train_config.downstream_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def build_encoder(model_config: ModelConfig) -> SignalEncoder:
    return SignalEncoder(
        in_channels=model_config.in_channels,
        cnn_channels=tuple(model_config.cnn_channels),
        cnn_kernel_sizes=tuple(model_config.cnn_kernel_sizes),
        cnn_strides=tuple(model_config.cnn_strides),
        transformer_layers=model_config.transformer_layers,
        transformer_dim=model_config.transformer_dim,
        transformer_heads=model_config.transformer_heads,
        transformer_ff_dim=model_config.transformer_ff_dim,
        transformer_dropout=model_config.transformer_dropout,
        max_seq_len=model_config.max_seq_len,
        pool_type=model_config.pool_type,
    )


def load_pretrained_encoder(
    checkpoint_path: str,
    model_config: ModelConfig,
    encoder_type: str,  # "context" or "target"
    device: torch.device,
) -> SignalEncoder:
    """Load a pre-trained encoder from JEPA checkpoint."""
    encoder = build_encoder(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    key = f"{encoder_type}_encoder"
    if key in ckpt:
        state_dict = ckpt[key]
    else:
        # Try model_state_dict
        msd = ckpt["model_state_dict"]
        prefix = f"{encoder_type}_encoder."
        state_dict = {
            k[len(prefix):]: v
            for k, v in msd.items()
            if k.startswith(prefix)
        }

    encoder.load_state_dict(state_dict, strict=True)
    print(f"Loaded {encoder_type}_encoder from {checkpoint_path}")
    return encoder


def train_epoch(
    model, dataloader, optimizer, criterion, device, is_dual: bool = False
):
    """Single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if is_dual:
            ecg, ppg, labels = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
        else:
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, is_dual: bool = False):
    """Evaluation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        if is_dual:
            ecg, ppg, labels = batch
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)
            logits = model(ecg, ppg)
        else:
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

        loss = criterion(logits, labels)

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return running_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def train_downstream(
    config: Config,
    checkpoint_path: str,
    modality: str = "ppg",  # "ecg", "ppg", or "dual"
):
    """
    Downstream fine-tuning pipeline.

    Args:
        config: master configuration
        checkpoint_path: path to pre-trained JEPA checkpoint
        modality: which signal to use for classification
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Modality: {modality}")

    is_dual = modality == "dual"

    # Data
    if is_dual:
        train_loader, test_loader = build_dual_dataloaders(config.data, config.train)
        # Dual uses both ECG (context) and PPG (target) encoders
        ecg_encoder = load_pretrained_encoder(
            checkpoint_path, config.model, "context", device
        )
        ppg_encoder = load_pretrained_encoder(
            checkpoint_path, config.model, "target", device
        )
        model = DualChannelClassifier(
            ecg_encoder=ecg_encoder,
            ppg_encoder=ppg_encoder,
            encoder_dim=config.model.transformer_dim,
            num_classes=config.data.num_classes,
        ).to(device)
    else:
        train_loader, test_loader = build_downstream_dataloaders(
            config.data, config.train, modality
        )
        # Single-channel: use target_encoder for PPG, context_encoder for ECG
        encoder_type = "target" if modality == "ppg" else "context"
        encoder = load_pretrained_encoder(
            checkpoint_path, config.model, encoder_type, device
        )
        model = SignalClassifier(
            encoder=encoder,
            encoder_dim=config.model.transformer_dim,
            num_classes=config.data.num_classes,
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Linear Probe ──
    print("\n=== Phase 1: Linear Probe (frozen encoder) ===")
    if is_dual:
        model.freeze_encoders()
    else:
        model.freeze_encoder()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.train.downstream_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.train.downstream_probe_epochs)

    for epoch in range(config.train.downstream_probe_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, is_dual
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device, is_dual
        )
        scheduler.step()

        print(
            f"Probe Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
        )

    # ── Phase 2: Full Fine-tune ──
    print("\n=== Phase 2: Full Fine-tune ===")
    if is_dual:
        model.unfreeze_encoders()
    else:
        model.unfreeze_encoder()

    full_epochs = config.train.downstream_epochs - config.train.downstream_probe_epochs
    optimizer = AdamW(model.parameters(), lr=config.train.downstream_lr * 0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=full_epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(full_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, is_dual
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device, is_dual
        )
        scheduler.step()

        print(
            f"FT Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_acc": test_acc,
            }

    # Save best model
    save_path = os.path.join(config.output_dir, f"downstream_{modality}_best.pt")
    torch.save(best_state, save_path)
    print(f"\nBest Test Acc: {best_acc:.2f}% → saved to {save_path}")

    return best_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained JEPA checkpoint")
    parser.add_argument("--modality", type=str, default="ppg",
                        choices=["ecg", "ppg", "dual"],
                        help="Which signal to use for classification")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    config = Config()
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    train_downstream(config, args.checkpoint, args.modality)
