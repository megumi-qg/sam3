#!/usr/bin/env python3
"""Find qualitative-display candidates for the CMPB manuscript.

The ranking favors test cases where SAM3-Scribble has a high per-case DSC and
beats all selected non-leakage comparison models by a clear margin.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from compute_statistical_tests import SOURCES, extract_per_case_metrics


OUT_DIR = Path("/home/gaoqi/sam3/gq_paper/cmpb/results_summary")
DATASETS = ["ACDC", "BTCV_cervix", "PROMISE12"]
OURS = "SAM3-Scribble"
BASELINES = ["SAT-Pro", "ScribFormer", "ScribbleBench", "EFFDNet"]


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary: dict[str, list[dict[str, object]]] = {}

    for dataset in DATASETS:
        case_metrics: dict[str, dict[str, float]] = {}
        for method in [OURS, *BASELINES]:
            path = Path(SOURCES[(method, dataset)])
            case_metrics[method] = {
                case_id: metrics["dice"]
                for case_id, metrics in extract_per_case_metrics(path).items()
                if "dice" in metrics
            }

        common = set(case_metrics[OURS])
        for method in BASELINES:
            common &= set(case_metrics[method])

        dataset_rows = []
        for case_id in sorted(common):
            ours = case_metrics[OURS][case_id]
            baseline_values = [case_metrics[method][case_id] for method in BASELINES]
            best_baseline = max(baseline_values)
            mean_baseline = sum(baseline_values) / len(baseline_values)
            worst_baseline = min(baseline_values)
            min_margin = ours - best_baseline
            mean_margin = ours - mean_baseline
            row = {
                "dataset": dataset,
                "case_id": case_id,
                "ours_dsc": ours,
                "best_baseline_dsc": best_baseline,
                "mean_baseline_dsc": mean_baseline,
                "worst_baseline_dsc": worst_baseline,
                "min_margin_vs_best_baseline": min_margin,
                "mean_margin_vs_baselines": mean_margin,
                "all_baselines_beaten": min_margin > 0,
                **{f"{method}_dsc": case_metrics[method][case_id] for method in BASELINES},
            }
            rows.append(row)
            dataset_rows.append(row)

        ranked = sorted(
            dataset_rows,
            key=lambda row: (
                not bool(row["all_baselines_beaten"]),
                -float(row["min_margin_vs_best_baseline"]),
                -float(row["ours_dsc"]),
                float(row["mean_baseline_dsc"]),
            ),
        )
        summary[dataset] = ranked[:10]

    csv_path = OUT_DIR / "qualitative_display_candidates.csv"
    fieldnames = [
        "dataset",
        "case_id",
        "ours_dsc",
        "best_baseline_dsc",
        "mean_baseline_dsc",
        "worst_baseline_dsc",
        "min_margin_vs_best_baseline",
        "mean_margin_vs_baselines",
        "all_baselines_beaten",
        *[f"{method}_dsc" for method in BASELINES],
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row[key]) if isinstance(row[key], float) else row[key] for key in fieldnames})

    json_path = OUT_DIR / "qualitative_display_candidates_top10.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for dataset in DATASETS:
        print(f"\n{dataset}: top qualitative candidates")
        print("case_id ours best_base mean_base gap_best SAT ScribFormer ScribbleBench EFFDNet")
        for row in summary[dataset][:5]:
            print(
                f"{row['case_id']} "
                f"{float(row['ours_dsc']):.4f} "
                f"{float(row['best_baseline_dsc']):.4f} "
                f"{float(row['mean_baseline_dsc']):.4f} "
                f"{float(row['min_margin_vs_best_baseline']):+.4f} "
                f"{float(row['SAT-Pro_dsc']):.4f} "
                f"{float(row['ScribFormer_dsc']):.4f} "
                f"{float(row['ScribbleBench_dsc']):.4f} "
                f"{float(row['EFFDNet_dsc']):.4f}"
            )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
