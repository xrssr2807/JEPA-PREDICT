import json
import pickle

import numpy as np
import torch

from official_fm_baselines.common import (
    DISEASE_LABELS,
    EmbeddingCache,
    PPGSegmentDataset,
    PatientEmbeddingDataset,
    label_vector,
    load_split_manifest,
)
from official_fm_baselines.extract_embeddings import _resample_batch
from official_fm_baselines.train_cached_mil import CachedEmbeddingMIL


def test_merged_other_disease_label():
    labels = label_vector({"脑卒中（中风）": 1, "冠心病": 1})
    assert labels.shape == (8,)
    assert labels[3] == 1
    assert labels[4] == 1


def test_split_is_patient_disjoint(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    names = ["train_a_0.pkl", "train_b_0.pkl", "test_c_0.pkl"]
    sample = {
        "data": np.ones((2, 1000), dtype=np.float32),
        "label": {},
        "sampling_rate": 100,
    }
    for name in names:
        with open(data_dir / name, "wb") as handle:
            pickle.dump(sample, handle)
    split = {"train": names[:2], "val": names[2:]}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    result = load_split_manifest(
        str(split_path), str(data_dir), expected_sha256=""
    )
    assert result == split


def test_patient_embedding_padding_and_mask():
    cache = EmbeddingCache(
        embeddings=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        labels=torch.zeros(3, len(DISEASE_LABELS)),
        uids=["a", "a", "b"],
        files=["a0", "a1", "b0"],
        metadata={},
    )
    dataset = PatientEmbeddingDataset(cache, max_segments=3, train=False)
    bag, labels, uid, mask = dataset[0]
    assert uid == "a"
    assert bag.shape == (3, 4)
    assert labels.shape == (8,)
    assert mask.tolist() == [True, True, False]


def test_polyphase_resampling_preserves_ten_second_duration():
    signal = torch.randn(2, 1, 1000)
    rates = torch.tensor([100.0, 100.0])
    output = _resample_batch(signal, rates, target_rate_hz=125.0)
    assert output.shape == (2, 1, 1250)
    assert torch.isfinite(output).all()


def test_pulseppg_resampling_preserves_ten_second_duration():
    signal = torch.randn(2, 1, 1000)
    rates = torch.tensor([100.0, 100.0])
    output = _resample_batch(signal, rates, target_rate_hz=50.0)
    assert output.shape == (2, 1, 500)
    assert torch.isfinite(output).all()


def test_frozen_cached_mil_checkpoint_round_trip(tmp_path):
    model = CachedEmbeddingMIL(input_dim=4, hidden_dim=8, num_classes=8, dropout=0.3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 4,
        "hidden_dim": 8,
        "labels": DISEASE_LABELS,
        "seed": 42,
        "test_set_used": False,
    }
    path = tmp_path / "best_validation_model.pt"
    torch.save(checkpoint, path)
    restored = CachedEmbeddingMIL(input_dim=4, hidden_dim=8, num_classes=8, dropout=0.3)
    restored.load_state_dict(torch.load(path, weights_only=False)["model_state_dict"], strict=True)


def test_test_split_requires_explicit_manifest_request(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    names = ["train_a_0.pkl", "val_b_0.pkl", "test_c_0.pkl"]
    sample = {"data": np.ones((2, 1000)), "label": {}, "sampling_rate": 100}
    for name in names:
        with open(data_dir / name, "wb") as handle:
            pickle.dump(sample, handle)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": names[:1], "val": names[1:2], "test": names[2:]}), encoding="utf-8")
    default = load_split_manifest(str(split_path), str(data_dir), expected_sha256="")
    assert "test" not in default
    explicit = load_split_manifest(str(split_path), str(data_dir), split_names=("test",), expected_sha256="")
    assert explicit["test"] == names[2:]
