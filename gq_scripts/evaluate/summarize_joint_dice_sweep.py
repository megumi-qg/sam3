#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def fmt(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    rows = []
    for ckpt_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        values = {}
        for dataset in ("acdc", "mscmr", "isbi"):
            path = ckpt_dir / dataset / f"evaluation_results_{dataset}.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            values[dataset] = data.get("overall", {}).get("dice_mean")
        if len(values) == 3:
            macro = sum(values.values()) / len(values)
            rows.append((macro, ckpt_dir.name, values))
        elif values:
            rows.append((float("-inf"), ckpt_dir.name, values))

    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out_path = root / "summary.tsv"
    with out_path.open("w") as f:
        f.write("rank\tcheckpoint\tmacro_dice\tacdc_dice\tmscmr_dice\tisbi_dice\n")
        rank = 0
        for macro, ckpt, values in rows:
            if math.isfinite(macro):
                rank += 1
                rank_text = str(rank)
                macro_text = fmt(macro)
            else:
                rank_text = "partial"
                macro_text = "nan"
            f.write(
                f"{rank_text}\t{ckpt}\t{macro_text}\t"
                f"{fmt(values.get('acdc'))}\t{fmt(values.get('mscmr'))}\t{fmt(values.get('isbi'))}\n"
            )
    print(out_path)
    if rows:
        print(out_path.read_text())


if __name__ == "__main__":
    main()
