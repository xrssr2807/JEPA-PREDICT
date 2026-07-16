"""
预处理脚本：批量将原始 .pkl 文件预处理为 .pt 文件。
- 提取 ECG (ch0) 和 PPG (ch4)
- 每通道 Z-score 归一化
- 保存为单个 .pt 文件，方便训练时快速加载
"""
import os
import pickle
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import numpy as np
import torch

from dataset.data import compute_signal_stats, infer_pretrain_uid


PREPROCESS_VERSION = 2


def zscore_per_channel(x: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Per-channel Z-score normalization with strict finite-value checks."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("raw signal contains NaN or Inf")

    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (x - mean) / std
    if clip > 0:
        normalized = np.clip(normalized, -clip, clip)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized signal contains NaN or Inf")
    return normalized.astype(np.float32, copy=False)


def process_one(args):
    """处理单个文件，返回 (success, filename, error_msg)"""
    src_path, dst_path, channels = args
    try:
        with open(src_path, "rb") as f:
            sample = pickle.load(f)

        data = np.asarray(sample["data"])  # (5, 3000)
        if data.ndim != 2 or max(channels) >= data.shape[0]:
            raise ValueError(
                f"invalid signal shape {data.shape} for channels {channels}"
            )
        data = data[channels]  # (2, 3000): [ECG, PPG]
        if not np.isfinite(data).all():
            raise ValueError("raw signal contains NaN or Inf")
        ecg_stats = torch.from_numpy(compute_signal_stats(data[0])).float()
        data = zscore_per_channel(data)  # 归一化

        ecg = torch.from_numpy(data[0:1].copy()).float()  # (1, 3000)
        ppg = torch.from_numpy(data[1:2].copy()).float()  # (1, 3000)

        if not all(torch.isfinite(tensor).all() for tensor in (ecg, ppg, ecg_stats)):
            raise ValueError("preprocessed tensors contain NaN or Inf")

        tmp_path = dst_path + ".tmp"
        torch.save(
            {
                "ecg": ecg,
                "ppg": ppg,
                "ecg_stats": ecg_stats,
                "uid": infer_pretrain_uid(os.path.basename(src_path)),
                "preprocess_version": PREPROCESS_VERSION,
            },
            tmp_path,
        )
        os.replace(tmp_path, dst_path)
        return True, src_path, None
    except Exception as e:
        tmp_path = dst_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # A failed overwrite must not leave a previously generated bad sample.
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False, src_path, str(e)


def main():
    parser = argparse.ArgumentParser(description="预处理 JEPA 训练数据")
    parser.add_argument("--src_dir", type=str, default="/root/autodl-tmp/split",
                        help="原始 .pkl 数据目录")
    parser.add_argument("--dst_dir", type=str, default="/root/autodl-tmp/split_processed",
                        help="预处理后 .pt 文件输出目录")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 4],
                        help="要提取的通道索引 (默认: 0 4 = ECG PPG)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行进程数")
    parser.add_argument("--skip_bad", action="store_true", default=True,
                        help="跳过损坏文件 (默认: True)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖旧 .pt 文件，用于升级stats等预处理字段")
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # 收集所有 .pkl 文件
    known_bad = {"combined_processed_data_2d_part10269_10.pkl"}
    all_files = sorted([
        f for f in os.listdir(args.src_dir)
        if f.endswith(".pkl") and f.startswith("combined_processed_data")
    ])
    if args.skip_bad:
        all_files = [f for f in all_files if f not in known_bad]

    print(f"找到 {len(all_files)} 个文件待处理")
    print(f"源目录: {args.src_dir}")
    print(f"输出目录: {args.dst_dir}")
    print(f"通道: {args.channels}, 并行进程: {args.workers}")

    # 构建任务列表
    tasks = []
    for fname in all_files:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.dst_dir, fname.replace(".pkl", ".pt"))
        if os.path.exists(dst) and not args.overwrite:
            continue  # 跳过已处理完成的
        tasks.append((src, dst, args.channels))

    if not tasks:
        print("所有文件已处理完毕，无需重复处理。")
        return

    print(f"实际待处理: {len(tasks)} 个文件（跳过 {len(all_files) - len(tasks)} 个已完成的）")

    # 多进程处理
    success_count = 0
    fail_count = 0
    failed_files = []

    with Pool(processes=args.workers) as pool:
        with tqdm(total=len(tasks), desc="预处理进度") as pbar:
            for ok, path, err in pool.imap_unordered(process_one, tasks):
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_files.append((path, err))
                pbar.update(1)
                pbar.set_postfix({"成功": success_count, "失败": fail_count})

    print(f"\n预处理完成: 成功 {success_count}, 失败 {fail_count}")
    if failed_files:
        print("失败文件（前10个）:")
        for path, err in failed_files[:10]:
            print(f"  {os.path.basename(path)}: {err}")

    # 保存文件列表
    processed_files = sorted([
        f for f in os.listdir(args.dst_dir) if f.endswith(".pt")
    ])
    print(f"输出目录共有 {len(processed_files)} 个 .pt 文件")


if __name__ == "__main__":
    main()
