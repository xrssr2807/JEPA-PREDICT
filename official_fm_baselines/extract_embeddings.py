import argparse
import json
import os
import time

import torch
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


def build_adapter(name: str, device: torch.device):
    if name == "moment_small":
        return MomentSmallAdapter(device)
    raise ValueError(f"Unsupported official model: {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(OFFICIAL_MODELS), required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
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
    adapter = build_adapter(args.model, device)
    model_spec = OFFICIAL_MODELS[args.model]

    summary = {
        "model": args.model,
        "official_checkpoint": model_spec["checkpoint"],
        "official_source": model_spec["source"],
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

