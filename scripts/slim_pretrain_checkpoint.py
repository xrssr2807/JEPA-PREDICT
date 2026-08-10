"""Create an encoder-only evaluation checkpoint after sealed downstream use."""

import argparse
import os

import torch


REQUIRED_KEYS = (
    "pretrain_phase",
    "context_encoder",
    "ppg_encoder",
    "phase2_config",
)

OPTIONAL_KEYS = (
    "epoch",
    "target_encoder",
    "val_loss",
    "val_metrics",
    "best_val_loss",
    "seed",
    "data_split_seed",
    "train_segments",
    "val_segments",
)


def build_slim_payload(checkpoint):
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(f"Cannot slim checkpoint; missing keys: {missing}")
    payload = {key: checkpoint[key] for key in REQUIRED_KEYS}
    payload.update({
        key: checkpoint[key]
        for key in OPTIONAL_KEYS
        if key in checkpoint
    })
    payload["checkpoint_format"] = "encoder_eval_slim_v1"
    payload["removed_training_state"] = True
    return payload


def slim_checkpoint(source, destination):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    payload = build_slim_payload(checkpoint)

    destination = os.path.abspath(destination)
    temporary = destination + ".tmp"
    torch.save(payload, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    if any(key not in verified for key in REQUIRED_KEYS):
        raise RuntimeError("Slim checkpoint verification failed")
    os.replace(temporary, destination)
    return os.path.getsize(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    destination = args.output or args.input
    size = slim_checkpoint(args.input, destination)
    print(
        f"[Slim] checkpoint={os.path.abspath(destination)} "
        f"size_bytes={size} format=encoder_eval_slim_v1"
    )


if __name__ == "__main__":
    main()
