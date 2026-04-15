#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按固定患者 ID 列表，将 nifti_split_img_gt_scribble_official/train 下各子文件夹中的样本
划分到 nifti_split_img_gt_scribble_100/{train,val,test}/<子文件夹>/。

患者编号与文件名前缀一致：1 -> patient001_*
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set

# 共 100 名患者，均来自原 ACDC 训练集划分
SPLIT: Dict[str, Set[int]] = {
    "train": {
        2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19, 20, 21, 23, 26, 28, 29,
        31, 33, 34, 35, 37, 38, 40, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55,
        56, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 74, 75, 76, 83, 85, 86,
        87, 88, 89, 90, 91, 92, 93, 94, 97, 98, 99, 100,
    },
    "val": {1, 17, 18, 22, 25, 27, 32, 44, 49, 57, 79, 81, 84, 95, 96},
    "test": {5, 10, 24, 30, 36, 39, 41, 60, 64, 72, 73, 77, 78, 80, 82},
}


def _patient_prefix(pid: int) -> str:
    return f"patient{pid:03d}_"


def _collect_subdirs(src_train: Path) -> List[Path]:
    return sorted([p for p in src_train.iterdir() if p.is_dir()])


def _files_for_patient(subdir: Path, pid: int) -> List[Path]:
    prefix = _patient_prefix(pid)
    return sorted(subdir.glob(f"{prefix}*.nii.gz"))


def run(
    src_train: Path,
    dst_root: Path,
    *,
    use_symlink: bool,
    subdirs: Iterable[str] | None,
) -> None:
    if subdirs is None:
        subdir_paths = _collect_subdirs(src_train)
    else:
        subdir_paths = [src_train / name for name in subdirs]
        for p in subdir_paths:
            if not p.is_dir():
                raise FileNotFoundError(f"源子目录不存在: {p}")

    # 校验划分互斥且并集为 1..100
    all_ids: Set[int] = set()
    for name, ids in SPLIT.items():
        overlap = all_ids & ids
        if overlap:
            raise ValueError(f"划分冲突 {name} 与已有 ID 重复: {sorted(overlap)}")
        all_ids |= ids
    if all_ids != set(range(1, 101)):
        missing = set(range(1, 101)) - all_ids
        extra = all_ids - set(range(1, 101))
        raise ValueError(f"划分须恰好覆盖 1..100，缺 {sorted(missing)} 多 {sorted(extra)}")

    pid_to_split = {}
    for split_name, ids in SPLIT.items():
        for pid in ids:
            pid_to_split[pid] = split_name

    for split_name in SPLIT:
        for sub in subdir_paths:
            (dst_root / split_name / sub.name).mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing = 0
    for sub in subdir_paths:
        for pid in range(1, 101):
            split_name = pid_to_split[pid]
            files = _files_for_patient(sub, pid)
            if not files:
                n_missing += 1
                print(f"[WARN] 未找到 {sub.name} 下患者 {pid} 的文件（前缀 {_patient_prefix(pid)}）")
                continue
            for src in files:
                dst = dst_root / split_name / sub.name / src.name
                if dst.exists():
                    dst.unlink()
                if use_symlink:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
                n_copied += 1

    print(f"完成：写入 {n_copied} 个文件；缺失患者-子目录组合数 {n_missing}（若无 WARN 则为 0）")
    print(f"目标根目录: {dst_root.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="按 100 患者划分复制/链接 ACDC NIfTI train 样本")
    ap.add_argument(
        "--src_train",
        type=Path,
        default=Path(
            "/home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_official/train"
        ),
    )
    ap.add_argument(
        "--dst_root",
        type=Path,
        default=Path("/home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100"),
    )
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="使用符号链接而非复制（节省空间，源路径需保持可用）",
    )
    ap.add_argument(
        "--subdirs",
        nargs="*",
        default=None,
        help="仅处理这些子文件夹名；默认处理源 train 下全部子目录",
    )
    args = ap.parse_args()
    run(args.src_train, args.dst_root, use_symlink=args.symlink, subdirs=args.subdirs)


if __name__ == "__main__":
    main()
