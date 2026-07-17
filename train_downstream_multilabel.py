"""Compatibility CLI for the canonical multidisease downstream pipeline.

All multilabel training is implemented in train_downstream.py so every entry
point uses the same patient-disjoint train/val/test manifest and evaluation.
"""
import argparse
import os

from config import Config
from train_downstream import train_downstream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/jepa_best.pt",
        help="Path to the pretrained JEPA checkpoint",
    )
    parser.add_argument("--output_dir", default="outputs_multidisease")
    parser.add_argument(
        "--multidisease_channel",
        choices=["both", "ppg", "ecg"],
        default=None,
    )
    parser.add_argument(
        "--multidisease_split",
        "--development_split",
        dest="multidisease_split",
        default=None,
        help="Patient-level train/val/test JSON manifest",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config()
    config.seed = args.seed
    config.output_dir = args.output_dir
    if args.multidisease_split is not None:
        config.data.multidisease_split_file = args.multidisease_split
    if args.multidisease_channel is not None:
        channel_map = {
            "both": "both",
            "ppg": str(config.data.multidisease_ppg_channel),
            "ecg": str(config.data.multidisease_ecg_channel),
        }
        config.data.multidisease_channel = channel_map[args.multidisease_channel]
        config.data.multidisease_dual_stream = args.multidisease_channel == "both"

    os.makedirs(config.output_dir, exist_ok=True)
    train_downstream(config, args.checkpoint, "multidisease")


if __name__ == "__main__":
    main()
