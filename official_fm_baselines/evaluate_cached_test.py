"""One-time evaluation of a frozen cached-MIL model on the sealed test split."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.losses import AsymmetricLoss
from official_fm_baselines.common import (
    DISEASE_LABELS,
    EmbeddingCache,
    PatientEmbeddingDataset,
    ensure_patient_counts,
    file_sha256,
    seed_everything,
)
from official_fm_baselines.train_cached_mil import CachedEmbeddingMIL, evaluate


AUTHORIZATION = "FINAL_ICASSP_2027"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_segments", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.authorization != AUTHORIZATION:
        raise RuntimeError(f"Authorization must equal {AUTHORIZATION}")

    checkpoint_path = os.path.join(args.run_dir, "best_validation_model.pt")
    validation_summary_path = os.path.join(args.run_dir, "summary.json")
    for path in (checkpoint_path, validation_summary_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(validation_summary_path, "r", encoding="utf-8") as handle:
        validation_summary = json.load(handle)
    if validation_summary.get("test_set_used", True):
        raise RuntimeError("Development summary does not certify a sealed test set")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_set_used", True):
        raise RuntimeError("Frozen checkpoint does not certify validation-only selection")
    if list(checkpoint.get("labels", [])) != DISEASE_LABELS:
        raise RuntimeError("Checkpoint label schema differs from the frozen eight-label schema")

    seed = int(checkpoint["seed"])
    seed_everything(seed)
    test_cache_path = os.path.join(args.cache_dir, "test_embeddings.pt")
    test_cache = EmbeddingCache.load(test_cache_path)
    ensure_patient_counts(test_cache, 1155, "test")
    if not test_cache.metadata.get("test_set_used", False):
        raise RuntimeError("Test cache is not explicitly marked as unsealed")
    if int(test_cache.embeddings.shape[-1]) != int(checkpoint["input_dim"]):
        raise RuntimeError("Frozen head and test embedding dimensions differ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatientEmbeddingDataset(test_cache, args.max_segments, train=False)
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = CachedEmbeddingMIL(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=len(DISEASE_LABELS),
        dropout=0.3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    metrics, labels, probabilities, uids = evaluate(model, loader, criterion, device)

    os.makedirs(args.output_dir, exist_ok=True)
    prediction_path = os.path.join(args.output_dir, "test_patient_predictions.npz")
    np.savez_compressed(
        prediction_path,
        uid=np.asarray(uids),
        labels=labels,
        probabilities=probabilities,
    )
    summary = {
        "seed": seed,
        "selection_source": "frozen validation checkpoint",
        "model_selection": validation_summary.get("model_selection"),
        "validation_metrics_at_selection": checkpoint.get("validation_metrics"),
        "test_metrics": metrics,
        "test_patients": len(uids),
        "test_set_used": True,
        "threshold_tuning_on_test": False,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_summary_sha256": file_sha256(validation_summary_path),
        "test_cache_sha256": file_sha256(test_cache_path),
    }
    with open(os.path.join(args.output_dir, "final_test_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "FINAL_TEST_COMPLETE"), "w", encoding="ascii") as handle:
        handle.write("test_set_used=true\nthreshold_tuning_on_test=false\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Complete] one-time sealed test evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
