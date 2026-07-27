"""Audit a multidisease manifest before running paper-facing experiments."""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import Config
from dataset.data import multidisease_label_value
from generate_multidisease_patient_split import uid_from_filename


def fail(message):
    raise ValueError(f"[SplitAudit] {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--data_dir", default=Config().data.multidisease_dir)
    parser.add_argument(
        "--skip_label_recount",
        action="store_true",
        help="Skip representative-patient label recount",
    )
    args = parser.parse_args()

    with open(args.split, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    metadata = manifest.get("metadata", {})
    expected_labels = list(Config().data.multidisease_labels)
    if metadata.get("disease_labels") != expected_labels:
        fail(
            "label schema mismatch; regenerate the split. "
            f"manifest={metadata.get('disease_labels')} config={expected_labels}"
        )

    if all(name in manifest for name in ("train", "val", "test")):
        split_names = ("train", "val", "test")
    elif all(
        name in manifest
        for name in ("feedback_train", "feedback_meta", "val", "test")
    ):
        split_names = ("feedback_train", "feedback_meta", "val", "test")
    else:
        fail("manifest must contain either 3-way or 4-way split roles")

    uid_sets = {}
    representative_files = {}
    all_files = set()
    for split_name in split_names:
        files = manifest[split_name]
        if not isinstance(files, list) or not files:
            fail(f"{split_name} must be a non-empty list")
        if len(files) != len(set(files)):
            fail(f"{split_name} contains duplicate filenames")
        uids = defaultdict(list)
        for filename in files:
            if (
                not isinstance(filename, str)
                or "/" in filename
                or "\\" in filename
                or not filename.endswith(".pkl")
            ):
                fail(f"invalid filename in {split_name}: {filename}")
            uids[uid_from_filename(filename)].append(filename)
        uid_sets[split_name] = set(uids)
        representative_files[split_name] = {
            uid: sorted(names)[0] for uid, names in uids.items()
        }
        all_files.update(files)

        expected_patients = metadata.get("patient_counts", {}).get(split_name)
        expected_files = metadata.get("file_counts", {}).get(split_name)
        if expected_patients is not None and expected_patients != len(uids):
            fail(
                f"{split_name} patient count mismatch: "
                f"metadata={expected_patients}, actual={len(uids)}"
            )
        if expected_files is not None and expected_files != len(files):
            fail(
                f"{split_name} file count mismatch: "
                f"metadata={expected_files}, actual={len(files)}"
            )

    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = uid_sets[left] & uid_sets[right]
            if overlap:
                fail(
                    f"{left}/{right} patient leakage: {len(overlap)} UIDs; "
                    f"examples={sorted(overlap)[:5]}"
                )

    missing = [
        filename
        for filename in sorted(all_files)
        if not os.path.isfile(os.path.join(args.data_dir, filename))
    ]
    if missing:
        fail(
            f"{len(missing)} manifest files are missing from {args.data_dir}; "
            f"examples={missing[:5]}"
        )

    if not args.skip_label_recount:
        recorded = metadata.get("positive_patient_counts", {})
        for split_name in split_names:
            recounted = {label: 0 for label in expected_labels}
            for filename in representative_files[split_name].values():
                with open(os.path.join(args.data_dir, filename), "rb") as handle:
                    sample = pickle.load(handle)
                labels = sample.get("label", {})
                for label in expected_labels:
                    recounted[label] += int(
                        multidisease_label_value(labels, label)
                    )
            if recorded.get(split_name) != recounted:
                fail(
                    f"{split_name} positive patient counts mismatch: "
                    f"metadata={recorded.get(split_name)}, actual={recounted}"
                )

    summary = {
        "split": os.path.abspath(args.split),
        "data_dir": os.path.abspath(args.data_dir),
        "roles": list(split_names),
        "patient_counts": {
            name: len(uid_sets[name]) for name in split_names
        },
        "file_counts": {
            name: len(manifest[name]) for name in split_names
        },
        "disease_labels": expected_labels,
        "uid_overlap": 0,
        "missing_files": 0,
        "label_recounted": not args.skip_label_recount,
        "status": "PASS",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
