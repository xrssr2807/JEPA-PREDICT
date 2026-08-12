import pickle

import numpy as np
import torch

from dataset.data import MultiDiseaseDataset, MultiDiseasePatientMILDataset
from models.encoder import SignalEncoder


def _write_segment(path, uid, value, samples=1000, rate=100.0):
    payload = {
        "uid": uid,
        "data": np.full((1, samples), value, dtype=np.float32),
        "sampling_rate": rate,
        "label": {"disease": 1},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_native_25hz_window_preserves_physical_duration(tmp_path):
    _write_segment(
        tmp_path / "train_patient_seg000.pkl",
        "patient",
        value=1.0,
        samples=1000,
        rate=100.0,
    )
    dataset = MultiDiseaseDataset(
        str(tmp_path),
        split="train",
        disease_labels=["disease"],
        normalize="none",
        canonical_sample_rate_hz=25.0,
    )
    signal, _, _ = dataset[0]
    assert signal.shape == (1, 250)


def test_native_25hz_external_window_maps_to_100hz_grid(tmp_path):
    _write_segment(
        tmp_path / "test_patient_seg000.pkl",
        "patient",
        value=1.0,
        samples=250,
        rate=25.0,
    )
    dataset = MultiDiseaseDataset(
        str(tmp_path),
        split="test",
        disease_labels=["disease"],
        normalize="none",
        canonical_sample_rate_hz=100.0,
    )
    signal, _, _ = dataset[0]
    assert signal.shape == (1, 1000)


def test_three_short_windows_form_one_30_second_token(tmp_path):
    files = []
    for index, value in enumerate((1.0, 2.0, 3.0)):
        name = f"train_patient_seg{index:03d}.pkl"
        files.append(name)
        _write_segment(
            tmp_path / name,
            "patient",
            value=value,
            samples=1000,
            rate=100.0,
        )
    dataset = MultiDiseasePatientMILDataset(
        str(tmp_path),
        split="train",
        disease_labels=["disease"],
        normalize="none",
        files=files,
        train=False,
        max_segments=2,
        canonical_sample_rate_hz=25.0,
        segment_token_seconds=30.0,
    )
    segments, labels, uid, mask = dataset[0]
    assert segments.shape == (2, 1, 750)
    assert mask.tolist() == [True, False]
    assert uid == "patient"
    assert labels.tolist() == [1.0]
    assert np.isclose(float(segments[0, 0, :250].mean()), 1.0, atol=2e-3)
    assert np.isclose(float(segments[0, 0, 250:500].mean()), 2.0, atol=3e-3)
    assert np.isclose(float(segments[0, 0, 500:].mean()), 3.0, atol=4e-3)


def test_encoder_accepts_native_25hz_10s_and_30s_inputs():
    encoder = SignalEncoder(
        cnn_channels=(8, 8, 8, 8),
        transformer_layers=1,
        transformer_dim=8,
        transformer_heads=2,
        transformer_ff_dim=16,
        transformer_dropout=0.0,
        max_seq_len=200,
    ).eval()
    with torch.no_grad():
        pooled_10s, tokens_10s = encoder(
            torch.randn(2, 1, 250), return_all=True
        )
        pooled_30s, tokens_30s = encoder(
            torch.randn(2, 1, 750), return_all=True
        )
    assert pooled_10s.shape == (2, 8)
    assert pooled_30s.shape == (2, 8)
    assert tokens_10s.shape[1] == 16
    assert tokens_30s.shape[1] == 47
