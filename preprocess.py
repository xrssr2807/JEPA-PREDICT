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


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Per-channel Z-score normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std


def process_one(args):
    """处理单个文件，返回 (success, filename, error_msg)"""
    src_path, dst_path, channels = args
    try:
        with open(src_path, "rb") as f:
            sample = pickle.load(f)

        data = sample["data"]  # (5, 3000)
        data = data[channels]  # (2, 3000): [ECG, PPG]
        data = zscore_per_channel(data)  # 归一化

        ecg = torch.from_numpy(data[0:1].copy()).float()  # (1, 3000)
        ppg = torch.from_numpy(data[1:2].copy()).float()  # (1, 3000)

        torch.save({"ecg": ecg, "ppg": ppg}, dst_path)
        return True, src_path, None
    except Exception as e:
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
        if os.path.exists(dst):
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
