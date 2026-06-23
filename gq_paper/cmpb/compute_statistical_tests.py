#!/usr/bin/env python3
"""Compute paired statistical tests for CMPB segmentation results.

The script reads existing evaluation JSON files, extracts per-case metrics,
and compares SAM3-Scribble against selected baselines with paired Wilcoxon
signed-rank tests. It writes machine-readable CSV files and a compact LaTeX
table for the supplementary material.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon


OUT_DIR = Path("/home/gaoqi/sam3/gq_paper/cmpb/results_summary")

SOURCES = {
    ("SAM3-Scribble", "ACDC"): "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_acdc_0.7/evaluation_results_acdc.json",
    ("SAM3-Scribble", "BTCV_cervix"): "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json",
    ("SAM3-Scribble", "PROMISE12"): "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json",
    ("SAM3-Full", "ACDC"): "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_acdc_0.7/evaluation_results_acdc.json",
    ("SAM3-Full", "BTCV_cervix"): "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json",
    ("SAM3-Full", "PROMISE12"): "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json",
    ("ScribFormer", "ACDC"): "/home/gaoqi/ScribFormer/model/ACDC_CMPB/scribformer_cmpb/scribble/test_eval_summary.json",
    ("ScribFormer", "BTCV_cervix"): "/home/gaoqi/ScribFormer/model/BTCV_CMPB/scribformer_cmpb/scribble/test_eval_summary.json",
    ("ScribFormer", "PROMISE12"): "/home/gaoqi/ScribFormer/model/PROMISE12_CMPB/scribformer_cmpb/scribble/test_eval_summary.json",
    ("EFFDNet", "ACDC"): "/home/gaoqi/EFFDNet/model/ACDC_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json",
    ("EFFDNet", "BTCV_cervix"): "/home/gaoqi/EFFDNet/model/BTCV_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json",
    ("EFFDNet", "PROMISE12"): "/home/gaoqi/EFFDNet/model/PROMISE12_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json",
    ("ScribbleBench", "ACDC"): "/home/gaoqi/dataset/nnU-Net/results/Dataset120_ACDCS_CMPB/test_eval_summary_sam3_metrics.json",
    ("ScribbleBench", "BTCV_cervix"): "/home/gaoqi/dataset/nnU-Net/results/Dataset121_BTCVS_CMPB/test_eval_summary_sam3_metrics.json",
    ("ScribbleBench", "PROMISE12"): "/home/gaoqi/dataset/nnU-Net/results/Dataset122_PROMISES_CMPB/test_eval_summary_sam3_metrics.json",
    ("SAT-Pro", "ACDC"): "/home/gaoqi/SAT/gq_dataset/ACDC/test/results_pro_cmpb/results_acdc_cmpb.json",
    ("SAT-Pro", "BTCV_cervix"): "/home/gaoqi/SAT/gq_dataset/BTCV/test/results_pro_cmpb/results_btcv_cmpb.json",
    ("SAT-Pro", "PROMISE12"): "/home/gaoqi/SAT/gq_dataset/PROMISE12/test/results_pro_cmpb/results_promise12_cmpb.json",
}

DATASETS = ["ACDC", "BTCV_cervix", "PROMISE12"]
BASELINES = ["ScribbleBench", "EFFDNet", "ScribFormer", "SAM3-Full", "SAT-Pro"]
METRICS = ["dice", "iou", "hd95", "nsd"]
PERCENT_METRICS = {"dice", "iou", "nsd"}
DATASET_LABELS = {
    "ACDC": "ACDC",
    "BTCV_cervix": "BTCV\\_cervix",
    "PROMISE12": "PROMISE12",
}
METRIC_LABELS = {
    "dice": "DSC",
    "iou": "IoU",
    "hd95": "HD95",
    "nsd": "NSD",
}


def normalize_case_id(case_id: str) -> str:
    case_id = str(case_id).strip()
    case_id = re.sub(r"\.nii(\.gz)?$", "", case_id)
    case_id = re.sub(r"\.npz$", "", case_id)
    case_id = re.sub(r"(?i)-image$", "", case_id)
    case_id = re.sub(r"(?i)^case", "", case_id)
    return case_id


def clean_number(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def extract_per_case_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    cases: dict[str, dict[str, float]] = {}

    per_patient = data.get("per_patient")
    if isinstance(per_patient, dict):
        for raw_case_id, record in per_patient.items():
            if not isinstance(record, dict):
                continue
            overall = record.get("overall", record)
            if not isinstance(overall, dict):
                continue
            metrics: dict[str, float] = {}
            for metric in METRICS:
                val = clean_number(overall.get(metric))
                if val is not None:
                    metrics[metric] = val
            if metrics:
                cases[normalize_case_id(raw_case_id)] = metrics
        return cases

    per_sample = data.get("per_sample")
    if isinstance(per_sample, list):
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in per_sample:
            if not isinstance(row, dict):
                continue
            raw_case_id = row.get("patient_frame") or row.get("case") or row.get("patient") or row.get("image_id")
            if raw_case_id is None:
                continue
            case_id = normalize_case_id(raw_case_id)
            for metric in METRICS:
                val = clean_number(row.get(metric))
                if val is not None:
                    grouped[case_id][metric].append(val)
        for case_id, metric_values in grouped.items():
            cases[case_id] = {
                metric: avg
                for metric, values in metric_values.items()
                if (avg := mean(values)) is not None
            }
        return cases

    raise ValueError(f"No per_patient or per_sample records found in {path}")


def wilcoxon_pvalue(improvements: list[float]) -> tuple[float, float]:
    non_zero = [x for x in improvements if abs(x) > 1e-12]
    if not non_zero:
        return 0.0, 1.0
    try:
        stat, p_value = wilcoxon(improvements, zero_method="wilcox", correction=False)
    except ValueError:
        return 0.0, 1.0
    return float(stat), float(p_value)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def fmt_float(value: float | None, decimals: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def fmt_csv_value(key: str, value: Any) -> Any:
    if not isinstance(value, float):
        return value if value is not None else ""
    if key.startswith("p_") or key == "wilcoxon_statistic":
        return f"{value:.10g}"
    return fmt_float(value)


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def fmt_delta(value: float, metric: str) -> str:
    if metric in PERCENT_METRICS:
        return f"{value * 100:.2f}"
    return f"{value:.2f}"


def compute_rows() -> list[dict[str, Any]]:
    loaded = {
        key: extract_per_case_metrics(Path(path))
        for key, path in SOURCES.items()
        if Path(path).exists()
    }

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        ours = loaded.get(("SAM3-Scribble", dataset))
        if not ours:
            continue
        for baseline in BASELINES:
            other = loaded.get((baseline, dataset))
            if not other:
                continue
            common_cases = sorted(set(ours) & set(other))
            for metric in METRICS:
                paired_cases = [
                    case_id
                    for case_id in common_cases
                    if metric in ours[case_id] and metric in other[case_id]
                ]
                if len(paired_cases) < 2:
                    continue
                ours_values = [ours[case_id][metric] for case_id in paired_cases]
                baseline_values = [other[case_id][metric] for case_id in paired_cases]
                if metric == "hd95":
                    improvements = [b - o for o, b in zip(ours_values, baseline_values)]
                else:
                    improvements = [o - b for o, b in zip(ours_values, baseline_values)]
                stat, p_value = wilcoxon_pvalue(improvements)
                sorted_improvements = sorted(improvements)
                mid = len(sorted_improvements) // 2
                if len(sorted_improvements) % 2:
                    median_improvement = sorted_improvements[mid]
                else:
                    median_improvement = (sorted_improvements[mid - 1] + sorted_improvements[mid]) / 2
                rows.append({
                    "dataset": dataset,
                    "comparison": f"SAM3-Scribble vs {baseline}",
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": len(paired_cases),
                    "ours_mean": mean(ours_values),
                    "baseline_mean": mean(baseline_values),
                    "mean_improvement": mean(improvements),
                    "median_improvement": median_improvement,
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "p_holm_by_metric": None,
                    "p_holm_all": None,
                    "significant_0_05_by_metric": None,
                    "significant_0_05_all": None,
                    "paired_cases": ";".join(paired_cases),
                })

    all_adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, adjusted in zip(rows, all_adjusted):
        row["p_holm_all"] = adjusted
        row["significant_0_05_all"] = adjusted < 0.05

    by_metric: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_metric[row["metric"]].append(idx)
    for metric, indices in by_metric.items():
        adjusted = holm_adjust([rows[idx]["p_value"] for idx in indices])
        for idx, p_adj in zip(indices, adjusted):
            rows[idx]["p_holm_by_metric"] = p_adj
            rows[idx]["significant_0_05_by_metric"] = p_adj < 0.05

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "comparison",
        "baseline",
        "metric",
        "n_pairs",
        "ours_mean",
        "baseline_mean",
        "mean_improvement",
        "median_improvement",
        "wilcoxon_statistic",
        "p_value",
        "p_holm_by_metric",
        "p_holm_all",
        "significant_0_05_by_metric",
        "significant_0_05_all",
        "paired_cases",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: fmt_csv_value(key, row.get(key))
                for key in fieldnames
            })


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    dice_rows = [row for row in rows if row["metric"] == "dice"]
    baseline_order = {name: idx for idx, name in enumerate(BASELINES)}
    dice_rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), baseline_order[row["baseline"]]))

    lines = [
        "% Generated by gq_paper/cmpb/compute_statistical_tests.py.",
        "",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Paired Wilcoxon signed-rank tests on per-case DSC scores. Positive $\\Delta$ indicates that SAM3-Scribble is better than the compared method. $p_{Holm}$ is corrected within the DSC family across all listed comparisons.}",
        "\\label{tab:supp_statistical_tests_dsc}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l l c c c c}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Comparison} & \\textbf{$n$} & \\textbf{$\\Delta$ DSC (pp)} & \\textbf{$p$} & \\textbf{$p_{Holm}$} \\\\",
        "\\midrule",
    ]
    for row in dice_rows:
        sig = "$^{*}$" if row["significant_0_05_by_metric"] else ""
        lines.append(
            f"{DATASET_LABELS[row['dataset']]} & SAM3-Scribble vs {row['baseline']} & "
            f"{row['n_pairs']} & {fmt_delta(row['mean_improvement'], 'dice')} & "
            f"{fmt_p(row['p_value'])} & {fmt_p(row['p_holm_by_metric'])}{sig} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
        "\\noindent Asterisks denote comparisons that remain significant at $\\alpha=0.05$ after Holm correction within the DSC family.",
        "",
    ])
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_rows()
    write_csv(args.out_dir / "statistical_tests_paired_wilcoxon.csv", rows)
    write_latex(args.out_dir / "generated_statistical_tests.tex", rows)

    dsc_significant = sum(
        1 for row in rows
        if row["metric"] == "dice" and row["significant_0_05_by_metric"]
    )
    dsc_total = sum(1 for row in rows if row["metric"] == "dice")
    print(f"Wrote {len(rows)} tests to {args.out_dir}")
    print(f"DSC significant after Holm-by-metric correction: {dsc_significant}/{dsc_total}")


if __name__ == "__main__":
    main()
