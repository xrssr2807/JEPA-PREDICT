import hashlib
import json
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DISEASE_LABELS = [
    "高血压",
    "高血糖",
    "高血脂",
    "其他疾病",
    "冠心病",
    "心律失常（房颤、频发早搏等）",
    "糖尿病",
    "颈动脉斑块",
]
MERGED_LABELS = {
    "其他疾病": ("下肢动脉硬化闭塞症", "脑卒中（中风）"),
}
EXPECTED_SPLIT_SHA256 = (
    "e3d458a8e88c75bf0b144be9bca5c7ecc87173f75425fe5cf0e2d77822cb9716"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_from_filename(filename: str) -> str:
    parts = os.path.basename(filename).split("_")
    if parts and parts[0] in {"train", "val", "test"} and len(parts) >= 3:
        return parts[1]
    return parts[0]


def label_vector(label_dict: dict) -> np.ndarray:
    values = []
    for label in DISEASE_LABELS:
        sources = (label, *MERGED_LABELS.get(label, ()))
        values.append(float(any(bool(label_dict.get(source, 0)) for source in sources)))
    return np.asarray(values, dtype=np.float32)


def load_split_manifest(
    split_path: str,
    data_dir: str,
    split_names: Sequence[str] = ("train", "val"),
    expected_sha256: str = EXPECTED_SPLIT_SHA256,
) -> Dict[str, List[str]]:
    actual_hash = file_sha256(split_path)
    if expected_sha256 and actual_hash != expected_sha256:
        raise RuntimeError(
            f"Split hash mismatch: expected={expected_sha256}, actual={actual_hash}"
        )
    with open(split_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest_labels = payload.get("metadata", {}).get("disease_labels")
    if manifest_labels and list(manifest_labels) != DISEASE_LABELS:
        raise RuntimeError(
            f"Eight-label schema mismatch: {manifest_labels} != {DISEASE_LABELS}"
        )

    result = {}
    uid_sets = {}
    for split_name in split_names:
        files = payload.get(split_name)
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"Split '{split_name}' is missing or empty")
        missing = [name for name in files if not os.path.isfile(os.path.join(data_dir, name))]
        if missing:
            raise FileNotFoundError(
                f"Split '{split_name}' references {len(missing)} missing files; "
                f"examples={missing[:5]}"
            )
        result[split_name] = sorted(files)
        uid_sets[split_name] = {uid_from_filename(name) for name in files}

    names = list(split_names)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = uid_sets[left] & uid_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Patient leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )
    return result


class PPGSegmentDataset(Dataset):
    """Read the exact PPG windows listed in the frozen split manifest."""

    def __init__(self, data_dir: str, files: Sequence[str]):
        self.data_dir = data_dir
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        filename = self.files[index]
        with open(os.path.join(self.data_dir, filename), "rb") as handle:
            sample = pickle.load(handle)
        signal = np.asarray(sample["data"], dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]
        signal = signal[0:1]
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True)
        std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
        signal = np.clip((signal - mean) / std, -10.0, 10.0)
        labels = label_vector(sample["label"])
        uid = str(sample.get("uid", uid_from_filename(filename)))
        sampling_rate = float(
            np.asarray(sample.get("sampling_rate", 100.0)).reshape(-1)[0]
        )
        return (
            torch.from_numpy(signal.copy()),
            torch.from_numpy(labels),
            uid,
            filename,
            sampling_rate,
        )


@dataclass
class EmbeddingCache:
    embeddings: torch.Tensor
    labels: torch.Tensor
    uids: List[str]
    files: List[str]
    metadata: dict

    @classmethod
    def load(cls, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"embeddings", "labels", "uids", "files", "metadata"}
        missing = required - set(payload)
        if missing:
            raise RuntimeError(f"Embedding cache is missing keys: {sorted(missing)}")
        return cls(**{key: payload[key] for key in required})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        torch.save(
            {
                "embeddings": self.embeddings,
                "labels": self.labels,
                "uids": self.uids,
                "files": self.files,
                "metadata": self.metadata,
            },
            temporary,
        )
        os.replace(temporary, path)


class PatientEmbeddingDataset(Dataset):
    """Construct fixed-size patient bags from cached segment embeddings."""

    def __init__(self, cache: EmbeddingCache, max_segments: int, train: bool):
        self.cache = cache
        self.max_segments = int(max_segments)
        self.train = bool(train)
        self.uid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for index, uid in enumerate(cache.uids):
            self.uid_to_indices[str(uid)].append(index)
        self.uids = sorted(self.uid_to_indices)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int):
        uid = self.uids[index]
        candidates = self.uid_to_indices[uid]
        if len(candidates) > self.max_segments:
            if self.train:
                positions = np.random.choice(
                    len(candidates), self.max_segments, replace=False
                )
            else:
                positions = np.linspace(
                    0, len(candidates) - 1, self.max_segments
                ).round().astype(int)
            chosen = [candidates[int(position)] for position in positions]
        else:
            chosen = list(candidates)
            if self.train:
                np.random.shuffle(chosen)

        labels = self.cache.labels[chosen]
        if not torch.all(labels == labels[0]):
            raise RuntimeError(f"Labels disagree between segments for UID {uid}")
        bag = self.cache.embeddings[chosen].float()
        valid = bag.shape[0]
        if valid < self.max_segments:
            padding = torch.zeros(
                self.max_segments - valid,
                bag.shape[-1],
                dtype=bag.dtype,
            )
            bag = torch.cat([bag, padding], dim=0)
        mask = torch.zeros(self.max_segments, dtype=torch.bool)
        mask[:valid] = True
        return bag, labels[0].float(), uid, mask


def ensure_patient_counts(cache: EmbeddingCache, expected: int, split_name: str) -> None:
    actual = len(set(cache.uids))
    if actual != expected:
        raise RuntimeError(
            f"{split_name} patient count mismatch: expected={expected}, actual={actual}"
        )

