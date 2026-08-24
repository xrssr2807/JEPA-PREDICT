import argparse
import importlib
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader

from official_fm_baselines.common import (
    EXPECTED_SPLIT_SHA256,
    EmbeddingCache,
    PPGSegmentDataset,
    file_sha256,
    load_split_manifest,
    seed_everything,
)


OFFICIAL_MODELS = {
    "moment_small": {
        "checkpoint": "AutonLab/MOMENT-1-small",
        "source": "https://github.com/moment-timeseries-foundation-model/moment",
        "modality": "PPG",
    },
    "papagei_s": {
        "checkpoint": "papagei_s.pt",
        "source": "https://github.com/Nokia-Bell-Labs/papagei-foundation-model",
        "modality": "PPG",
        "target_rate_hz": 125.0,
    },
    "normwear": {
        "checkpoint": "normwear_pretrain_ckpt.pth",
        "source": "https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear",
        "modality": "PPG",
        "target_rate_hz": 64.0,
    },
    "units_x128": {
        "checkpoint": "units_x128_pretrain_checkpoint.pth",
        "source": "https://github.com/mims-harvard/UniTS",
        "modality": "generic univariate time series",
        "feature_protocol": "shared backbone only; no dataset-specific prompt",
    },
}


class MomentSmallAdapter:
    def __init__(self, device: torch.device):
        from momentfm import MOMENTPipeline

        self.model = MOMENTPipeline.from_pretrained(
            OFFICIAL_MODELS["moment_small"]["checkpoint"],
            model_kwargs={"task_name": "embedding"},
        )
        self.model.init()
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, signal: torch.Tensor, sampling_rate: torch.Tensor):
        del sampling_rate  # MOMENT consumes normalized samples, not a rate token.
        signal = signal.to(self.device, non_blocking=True)
        input_mask = torch.ones(
            signal.shape[0], signal.shape[-1], device=self.device
        )
        output = self.model(x_enc=signal, input_mask=input_mask)
        embeddings = output.embeddings
        if embeddings.ndim != 2:
            embeddings = embeddings.flatten(start_dim=1)
        return embeddings.float()


def _resample_batch(
    signal: torch.Tensor,
    sampling_rate: torch.Tensor,
    target_rate_hz: float,
) -> torch.Tensor:
    """Resample each rate group with PaPaGei's polyphase method."""
    source = signal.detach().cpu().numpy().astype(np.float32, copy=False)
    rates = sampling_rate.detach().cpu().numpy().astype(np.float64, copy=False)
    output = []
    for row, rate in zip(source, rates):
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid sampling rate: {rate}")
        ratio = Fraction(float(target_rate_hz) / float(rate)).limit_denominator(1000)
        resampled = resample_poly(row, ratio.numerator, ratio.denominator, axis=-1)
        output.append(np.asarray(resampled, dtype=np.float32))
    lengths = {item.shape[-1] for item in output}
    if len(lengths) != 1:
        raise RuntimeError(f"Mixed resampled lengths in one batch: {sorted(lengths)}")
    return torch.from_numpy(np.stack(output, axis=0))


class PaPaGeiSmallAdapter:
    def __init__(self, device: torch.device, official_repo: str, checkpoint: str):
        if not official_repo or not os.path.isdir(official_repo):
            raise FileNotFoundError(f"PaPaGei official repository not found: {official_repo}")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"PaPaGei checkpoint not found: {checkpoint}")
        if os.path.getsize(checkpoint) < 20_000_000:
            raise RuntimeError(
                f"PaPaGei checkpoint appears incomplete: {os.path.getsize(checkpoint)} bytes"
            )
        sys.path.insert(0, os.path.abspath(official_repo))
        from models.resnet import ResNet1DMoE

        self.model = ResNet1DMoE(
            in_channels=1,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=512,
            n_experts=3,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.device = device
        self.target_rate_hz = OFFICIAL_MODELS["papagei_s"]["target_rate_hz"]

    @torch.inference_mode()
    def __call__(self, signal: torch.Tensor, sampling_rate: torch.Tensor):
        signal = _resample_batch(signal, sampling_rate, self.target_rate_hz)
        signal = signal.to(self.device, non_blocking=True)
        embeddings = self.model(signal)[-1]
        if embeddings.ndim != 2:
            embeddings = embeddings.flatten(start_dim=1)
        return embeddings.float()


class NormWearAdapter:
    def __init__(self, device: torch.device, official_repo: str, checkpoint: str):
        if not official_repo or not os.path.isdir(official_repo):
            raise FileNotFoundError(f"NormWear official repository not found: {official_repo}")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"NormWear checkpoint not found: {checkpoint}")
        if os.path.getsize(checkpoint) < 500_000_000:
            raise RuntimeError(
                f"NormWear checkpoint appears incomplete: {os.path.getsize(checkpoint)} bytes"
            )
        repo = os.path.abspath(official_repo)
        package_parent = os.path.dirname(repo)
        package_name = os.path.basename(repo)
        sys.path.insert(0, package_parent)
        module = importlib.import_module(f"{package_name}.main_model")
        self.model = module.NormWearModel(
            weight_path=checkpoint,
            optimized_cwt=True,
        )
        self.model.to(device).eval()
        self.device = device
        self.target_rate_hz = OFFICIAL_MODELS["normwear"]["target_rate_hz"]

    @torch.inference_mode()
    def __call__(self, signal: torch.Tensor, sampling_rate: torch.Tensor):
        signal = _resample_batch(signal, sampling_rate, self.target_rate_hz)
        output = self.model.get_embedding(
            signal,
            sampling_rate=int(self.target_rate_hz),
            device=self.device,
        )
        if output.ndim != 4:
            raise RuntimeError(f"Unexpected NormWear embedding shape: {tuple(output.shape)}")
        return output.mean(dim=(1, 2)).float()


class UniTSX128Adapter:
    def __init__(self, device: torch.device, official_repo: str, checkpoint: str):
        if not official_repo or not os.path.isdir(official_repo):
            raise FileNotFoundError(f"UniTS official repository not found: {official_repo}")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"UniTS checkpoint not found: {checkpoint}")
        if os.path.getsize(checkpoint) < 90_000_000:
            raise RuntimeError(
                f"UniTS checkpoint appears incomplete: {os.path.getsize(checkpoint)} bytes"
            )
        sys.path.insert(0, os.path.abspath(official_repo))
        module = importlib.import_module("models.UniTS")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload["args"]
        config = [
            (
                "PPG_external",
                {
                    "dataset": "PPG_external",
                    "enc_in": 1,
                    "task_name": "classification",
                    "num_class": 1,
                },
            )
        ]
        self.model = module.Model(args, config, pretrain=False)
        state = payload.get("student", payload)
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        shared_prefixes = (
            "patch_embeddings.",
            "position_embedding.",
            "prompt2forecat.",
            "blocks.",
            "cls_head.",
            "forecast_head.",
        )
        shared_state = {
            key: value for key, value in state.items() if key.startswith(shared_prefixes)
        }
        incompatible = self.model.load_state_dict(shared_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected UniTS checkpoint keys: {unexpected[:10]}")
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, signal: torch.Tensor, sampling_rate: torch.Tensor):
        del sampling_rate  # UniTS has no physical-rate input.
        x = signal.to(self.device, non_blocking=True).transpose(1, 2)
        x, _, _, n_vars, _ = self.model.tokenize(x)
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])
        x = x + self.model.position_embedding(x)
        seq_len = x.shape[-2]
        x = self.model.backbone(x, prefix_len=0, seq_len=seq_len)
        return x.mean(dim=(1, 2)).float()


def build_adapter(
    name: str,
    device: torch.device,
    official_repo: str = "",
    checkpoint: str = "",
):
    if name == "moment_small":
        return MomentSmallAdapter(device)
    if name == "papagei_s":
        return PaPaGeiSmallAdapter(device, official_repo, checkpoint)
    if name == "normwear":
        return NormWearAdapter(device, official_repo, checkpoint)
    if name == "units_x128":
        return UniTSX128Adapter(device, official_repo, checkpoint)
    raise ValueError(f"Unsupported official model: {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(OFFICIAL_MODELS), required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--official_repo", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--expected_split_sha256", default=EXPECTED_SPLIT_SHA256)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_lists = load_split_manifest(
        args.split,
        args.data_dir,
        split_names=("train", "val"),
        expected_sha256=args.expected_split_sha256,
    )
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    adapter = build_adapter(
        args.model,
        device,
        official_repo=args.official_repo,
        checkpoint=args.checkpoint,
    )
    model_spec = OFFICIAL_MODELS[args.model]

    summary = {
        "model": args.model,
        "official_checkpoint": model_spec["checkpoint"],
        "official_source": model_spec["source"],
        "checkpoint_path": os.path.abspath(args.checkpoint) if args.checkpoint else "",
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else "",
        "split_sha256": file_sha256(args.split),
        "test_set_used": False,
        "splits": {},
    }
    for split_name in ("train", "val"):
        output_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
        if os.path.isfile(output_path):
            cache = EmbeddingCache.load(output_path)
            summary["splits"][split_name] = cache.metadata
            print(f"[Skip] existing cache: {output_path}")
            continue
        files = split_lists[split_name]
        if args.max_files > 0:
            files = files[:args.max_files]
        dataset = PPGSegmentDataset(args.data_dir, files)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        embedding_chunks = []
        label_chunks = []
        uids = []
        filenames = []
        started = time.time()
        for batch_index, (signal, labels, uid, filename, sampling_rate) in enumerate(loader):
            embeddings = adapter(signal, sampling_rate)
            if not torch.isfinite(embeddings).all():
                raise RuntimeError(
                    f"Non-finite embedding at {split_name} batch {batch_index}"
                )
            embedding_chunks.append(embeddings.cpu().half())
            label_chunks.append(labels.cpu().float())
            uids.extend([str(value) for value in uid])
            filenames.extend([str(value) for value in filename])
            if (batch_index + 1) % 100 == 0:
                print(
                    f"[Extract] {split_name} batches={batch_index + 1}/{len(loader)} "
                    f"segments={len(uids)}"
                )
        metadata = {
            "split": split_name,
            "segments": len(uids),
            "patients": len(set(uids)),
            "embedding_dim": int(embedding_chunks[0].shape[-1]),
            "dtype": "float16",
            "elapsed_seconds": time.time() - started,
            "test_set_used": False,
        }
        cache = EmbeddingCache(
            embeddings=torch.cat(embedding_chunks, dim=0),
            labels=torch.cat(label_chunks, dim=0),
            uids=uids,
            files=filenames,
            metadata=metadata,
        )
        cache.save(output_path)
        summary["splits"][split_name] = metadata
        print(f"[Saved] {output_path} {metadata}")

    with open(os.path.join(output_dir, "embedding_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "EMBEDDINGS_COMPLETE"), "w", encoding="ascii") as handle:
        handle.write("test_set_used=false\n")
    print(f"[Complete] official embeddings: {output_dir}")


if __name__ == "__main__":
    main()
