import pickle

import numpy as np
import pytest

from config import Config
from train_downstream import (
    build_external_multidisease_loader,
    evaluate_external_multidisease_checkpoint,
)


def _write_window(path, uid, labels, valid=None):
    payload = {
        "uid": uid,
        "data": np.linspace(-1.0, 1.0, 1000, dtype=np.float32)[None, :],
        "sampling_rate": 100,
        "label": labels,
        "label_valid": valid or {name: 1 for name in labels},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _external_config():
    config = Config()
    config.data.multidisease_channel = str(
        config.data.multidisease_ppg_channel
    )
    config.train.dataloader_workers = 0
    config.train.multidisease_mil_batch_size = 2
    return config


def test_external_loader_is_patient_level_and_deterministic(tmp_path):
    config = _external_config()
    labels = {name: 0 for name in config.data.multidisease_labels}
    labels["冠心病"] = 1
    labels["其他疾病"] = 1
    _write_window(tmp_path / "test_mcabc_seg000.pkl", "mcabc", labels)
    _write_window(tmp_path / "test_mcabc_seg001.pkl", "mcabc", labels)

    loader, dataset, provenance = build_external_multidisease_loader(
        str(tmp_path), config
    )
    segments, targets, uids, mask = next(iter(loader))

    assert len(dataset) == 1
    assert provenance["window_count"] == 2
    assert provenance["patient_count"] == 1
    assert provenance["positive_patient_counts"]["冠心病"] == 1
    assert provenance["positive_patient_counts"]["其他疾病"] == 1
    assert list(uids) == ["mcabc"]
    assert tuple(segments.shape) == (1, 8, 1, 1000)
    assert int(mask.sum()) == 2
    assert int(targets[0, config.train.chd_label_index]) == 1


def test_external_loader_respects_native_25hz_clock(tmp_path):
    config = _external_config()
    labels = {name: 0 for name in config.data.multidisease_labels}
    path = tmp_path / "test_mc25_seg000.pkl"
    payload = {
        "uid": "mc25",
        "data": np.linspace(-1.0, 1.0, 250, dtype=np.float32)[None, :],
        "sampling_rate": 25,
        "label": labels,
        "label_valid": {name: 1 for name in labels},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    loader, _, provenance = build_external_multidisease_loader(
        str(tmp_path), config
    )
    segments, _, _, mask = next(iter(loader))
    assert tuple(segments.shape) == (1, 8, 1, 1000)
    assert int(mask.sum()) == 1
    assert provenance["sampling_rates_hz"] == [25.0]
    assert provenance["window_duration_seconds"] == {"min": 10.0, "max": 10.0}


def test_external_loader_rejects_incomplete_labels(tmp_path):
    config = _external_config()
    labels = {name: 0 for name in config.data.multidisease_labels}
    valid = {name: 1 for name in labels}
    valid["冠心病"] = 0
    _write_window(
        tmp_path / "test_mcabc_seg000.pkl", "mcabc", labels, valid=valid
    )

    with pytest.raises(ValueError, match="incomplete label-valid masks"):
        build_external_multidisease_loader(str(tmp_path), config)


def test_external_evaluation_requires_internal_thresholds():
    config = _external_config()
    with pytest.raises(ValueError, match="never be used to tune thresholds"):
        evaluate_external_multidisease_checkpoint(
            model=None,
            saved_state={"model_state_dict": {}},
            external_loader=None,
            criterion=None,
            device=None,
            config=config,
            use_amp=False,
            provenance={},
        )
