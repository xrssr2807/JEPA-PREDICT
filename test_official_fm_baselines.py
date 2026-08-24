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

