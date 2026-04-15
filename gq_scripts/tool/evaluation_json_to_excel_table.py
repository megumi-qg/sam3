#!/usr/bin/env python3
"""
从 evaluation_results_*.json 中提取 per_class 与 overall 的 Dice、IoU、HD95、NSD
的 **均值与标准差**（若 JSON 含 ``*_std`` 字段则写入对应列；缺失则 std 列为空），
格式化为四位小数，输出制表符分隔文本，可直接全选复制粘贴到 Excel / WPS 表格。

说明：评估管线里与 ``batch_evaluate.py`` 一致地记录的是 **标准差**（std），
不是数学意义上的方差（var = std²）。若需要方差，可在表格中对 std 列自行平方。

输入 JSON 需包含顶层键：
  - per_class: { 类别名: { dice_mean, dice_std, iou_mean, ... }, ... }
  - overall: { dice_mean, dice_std, ... }

用法：
  python gq_scripts/tool/evaluation_json_to_excel_table.py <json路径>

默认将 TSV 写入与 JSON 同目录：<json 主文件名>_excel_table.tsv
  python gq_scripts/tool/evaluation_json_to_excel_table.py <json路径> -o /其它路径/summary.tsv

使用 --stdout 时仍将表格打印到标准输出（便于复制）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# (mean_key, std_key) 与 batch_evaluate 写入的 per_class / overall 字段一致
METRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("dice_mean", "dice_std"),
    ("iou_mean", "iou_std"),
    ("hd95_mean", "hd95_std"),
    ("nsd_mean", "nsd_std"),
)
HEADER: tuple[str, ...] = (
    "Class",
    "Dice_mean",
    "Dice_std",
    "IoU_mean",
    "IoU_std",
    "HD95_mean",
    "HD95_std",
    "NSD_mean",
    "NSD_std",
)


def _fmt4(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def _cells_mean_std(block: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for mean_key, std_key in METRIC_PAIRS:
        out.append(_fmt4(block.get(mean_key)))
        out.append(_fmt4(block.get(std_key)))
    return out


def extract_rows(data: dict[str, Any]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    per_class = data.get("per_class") or {}
    if not isinstance(per_class, dict):
        raise ValueError("顶层缺少有效的 per_class 对象")

    for class_name in per_class:
        block = per_class[class_name]
        if not isinstance(block, dict):
            continue
        cells = _cells_mean_std(block)
        rows.append((str(class_name), *cells))

    overall = data.get("overall")
    if overall is not None:
        if not isinstance(overall, dict):
            raise ValueError("overall 必须是对象")
        cells = _cells_mean_std(overall)
        rows.append(("Overall", *cells))

    return rows


def to_tsv(rows: list[tuple[str, ...]]) -> str:
    lines = ["\t".join(HEADER)]
    for r in rows:
        lines.append("\t".join(r))
    return "\n".join(lines) + "\n"


def default_output_path(json_path: Path) -> Path:
    return json_path.parent / f"{json_path.stem}_excel_table.tsv"


def main() -> int:
    p = argparse.ArgumentParser(description="将 evaluation JSON 转为可粘贴 Excel 的 TSV 表")
    p.add_argument("json_path", type=Path, help="evaluation_results_*.json 路径")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 TSV 路径（UTF-8）；默认与 JSON 同目录：<stem>_excel_table.tsv",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="同时将表格打印到标准输出",
    )
    args = p.parse_args()

    path = args.json_path
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("JSON 根节点必须是对象", file=sys.stderr)
        return 1

    try:
        rows = extract_rows(data)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if not rows:
        print("未提取到任何行（检查 per_class）", file=sys.stderr)
        return 1

    text = to_tsv(rows)
    out_path = args.output if args.output is not None else default_output_path(path.resolve())
    out_path.write_text(text, encoding="utf-8")
    print(f"已写入: {out_path}", file=sys.stderr)
    if args.stdout:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
