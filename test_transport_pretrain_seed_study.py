import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from config import Config
from train_pretrain import _global_pretrain_lr_factor
from scripts.summarize_transport_pretrain_seed_study import (
    _paired_bootstrap,
    _split_content_hash,
)


def test_pretrain_split_seed_is_independent_from_optimization_seed():
    config = Config()
    config.seed = 3407

    assert config.seed == 3407
    assert config.pretrain_split_seed == 42
    assert config.train.pretrain_checkpoint_interval == 20


def test_global_resume_lr_uses_original_absolute_schedule():
    steps_per_epoch = 271
    total_steps = 80 * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    assert _global_pretrain_lr_factor(0, warmup_steps, total_steps) == 1e-6
    assert _global_pretrain_lr_factor(
        warmup_steps, warmup_steps, total_steps
    ) == 1.0
    resumed_factor = _global_pretrain_lr_factor(
        76 * steps_per_epoch,
        warmup_steps,
        total_steps,
    )
    assert 0.0 < resumed_factor < 0.02
    assert _global_pretrain_lr_factor(
        total_steps, warmup_steps, total_steps
    ) == 0.0


def test_split_content_hash_ignores_metadata_and_file_order():
    first = {
        "optimization_seed": 42,
        "data_split_seed": 42,
        "train_files": ["b.pt", "a.pt"],
        "val_files": ["d.pt", "c.pt"],
        "train_uids": ["u2", "u1"],
        "val_uids": ["u4", "u3"],
    }
    second = {
        "optimization_seed": 2026,
        "data_split_seed": 42,
        "train_files": ["a.pt", "b.pt"],
        "val_files": ["c.pt", "d.pt"],
        "train_uids": ["u1", "u2"],
        "val_uids": ["u3", "u4"],
        "extra_metadata": "does not affect patient membership",
    }
    suffix = uuid4().hex
    root = Path(__file__).resolve().parent
    first_path = root / f".split_hash_first_{suffix}.json"
    second_path = root / f".split_hash_second_{suffix}.json"
    try:
        first_path.write_text(json.dumps(first), encoding="utf-8")
        second_path.write_text(json.dumps(second), encoding="utf-8")
        assert _split_content_hash(first_path) == _split_content_hash(second_path)
    finally:
        first_path.unlink(missing_ok=True)
        second_path.unlink(missing_ok=True)


def test_paired_bootstrap_detects_positive_transport_delta():
    labels = np.asarray(
        [[0], [0], [0], [0], [1], [1], [1], [1]],
        dtype=np.float64,
    )
    on_probabilities = np.asarray(
        [[0.05], [0.10], [0.15], [0.20], [0.80], [0.85], [0.90], [0.95]],
        dtype=np.float64,
    )
    off_probabilities = np.asarray(
        [[0.10], [0.70], [0.20], [0.60], [0.30], [0.80], [0.40], [0.90]],
        dtype=np.float64,
    )

    result = _paired_bootstrap(
        labels,
        on_probabilities,
        off_probabilities,
        chd_index=0,
        iterations=500,
        seed=2027,
    )

    assert result["valid_bootstrap_iterations"] > 400
    assert result["delta_chd_auc_ci95_low"] >= 0.0
    assert result["delta_macro_auc_ci95_low"] >= 0.0
