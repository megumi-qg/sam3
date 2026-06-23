#!/usr/bin/env python3
"""Aggregate CMPB comparison metrics from model-specific result JSON files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


OUT_DIR = Path("/home/gaoqi/sam3/gq_paper/cmpb/results_summary")


SOURCES = [
    # SAM3
    ("SAM3-Scribble", "ACDC", "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_acdc_0.7/evaluation_results_acdc.json"),
    ("SAM3-Scribble", "BTCV_cervix", "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json"),
    ("SAM3-Scribble", "PROMISE12", "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json"),
    ("SAM3-Full", "ACDC", "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_acdc_0.7/evaluation_results_acdc.json"),
    ("SAM3-Full", "BTCV_cervix", "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json"),
    ("SAM3-Full", "PROMISE12", "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json"),
    # Weakly supervised baselines
    ("ScribFormer", "ACDC", "/home/gaoqi/ScribFormer/model/ACDC_CMPB/scribformer_cmpb/scribble/test_eval_summary.json"),
    ("ScribFormer", "BTCV_cervix", "/home/gaoqi/ScribFormer/model/BTCV_CMPB/scribformer_cmpb/scribble/test_eval_summary.json"),
    ("ScribFormer", "PROMISE12", "/home/gaoqi/ScribFormer/model/PROMISE12_CMPB/scribformer_cmpb/scribble/test_eval_summary.json"),
    ("EFFDNet", "ACDC", "/home/gaoqi/EFFDNet/model/ACDC_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json"),
    ("EFFDNet", "BTCV_cervix", "/home/gaoqi/EFFDNet/model/BTCV_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json"),
    ("EFFDNet", "PROMISE12", "/home/gaoqi/EFFDNet/model/PROMISE12_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json"),
    ("ScribbleBench", "ACDC", "/home/gaoqi/dataset/nnU-Net/results/Dataset120_ACDCS_CMPB/test_eval_summary_sam3_metrics.json"),
    ("ScribbleBench", "BTCV_cervix", "/home/gaoqi/dataset/nnU-Net/results/Dataset121_BTCVS_CMPB/test_eval_summary_sam3_metrics.json"),
    ("ScribbleBench", "PROMISE12", "/home/gaoqi/dataset/nnU-Net/results/Dataset122_PROMISES_CMPB/test_eval_summary_sam3_metrics.json"),
    # Promptable foundation models
    ("BiomedParse v2", "ACDC", "/home/gaoqi/BiomedParse/gq_data/acdc/test/eval_results/acdc_v2_cmpb.json"),
    ("BiomedParse v2", "BTCV_cervix", "/home/gaoqi/BiomedParse/gq_data/btcv/test/eval_results/btcv_cervix_v2_cmpb.json"),
    ("BiomedParse v2", "PROMISE12", "/home/gaoqi/BiomedParse/gq_data/promise12/test/eval_results/promise12_v2_cmpb.json"),
    ("VoxTell", "ACDC", "/home/gaoqi/VoxTell/gq_result/acdc/cmpb_test/evaluation_results_acdc_cmpb.json"),
    ("VoxTell", "BTCV_cervix", "/home/gaoqi/VoxTell/gq_result/btcv/cmpb_test/evaluation_results_btcv_cmpb.json"),
    ("VoxTell", "PROMISE12", "/home/gaoqi/VoxTell/gq_result/promise12/cmpb_test/evaluation_results_promise12_cmpb.json"),
    ("SAT-Pro", "ACDC", "/home/gaoqi/SAT/gq_dataset/ACDC/test/results_pro_cmpb/results_acdc_cmpb.json"),
    ("SAT-Pro", "BTCV_cervix", "/home/gaoqi/SAT/gq_dataset/BTCV/test/results_pro_cmpb/results_btcv_cmpb.json"),
    ("SAT-Pro", "PROMISE12", "/home/gaoqi/SAT/gq_dataset/PROMISE12/test/results_pro_cmpb/results_promise12_cmpb.json"),
]


METHOD_ORDER = {
    "ScribbleBench": 1,
    "EFFDNet": 2,
    "ScribFormer": 3,
    "BiomedParse v2": 4,
    "VoxTell": 5,
    "SAT-Pro": 6,
    "SAM3-Full": 7,
    "SAM3-Scribble": 8,
}
DATASET_ORDER = {"ACDC": 1, "BTCV_cervix": 2, "PROMISE12": 3}
METRICS = ["iou", "dice", "hd95", "nsd"]
TEXT_PROMPTED_METHODS = ["BiomedParse v2", "VoxTell", "SAT-Pro"]
STATIC_SCRIBBLE_METHODS = ["ScribFormer", "ScribbleBench", "EFFDNet"]
OURS_METHODS = ["SAM3-Scribble", "SAM3-Full"]
DATASET_LABELS = {
    "ACDC": "ACDC",
    "BTCV_cervix": "BTCV\\_cervix",
    "PROMISE12": "PROMISE12",
}
DATASET_TABLE_LABELS = {
    "ACDC": "acdc",
    "BTCV_cervix": "btcv",
    "PROMISE12": "promise12",
}
STRUCTURE_ORDER = {
    "ACDC": ["left ventricle", "myocardium", "right ventricle"],
    "BTCV_cervix": ["bladder", "rectum", "small bowel", "uterus"],
    "PROMISE12": ["prostate"],
}
LEAKAGE_METHODS_BY_DATASET = {
    "ACDC": {"BiomedParse v2", "VoxTell", "SAT-Pro"},
    "BTCV_cervix": {"BiomedParse v2", "VoxTell"},
    "PROMISE12": {"BiomedParse v2", "VoxTell"},
}
ROUND_DECIMALS = {
    "iou": 4,
    "dice": 4,
    "hd95": 2,
    "nsd": 4,
}


def clean_number(x: Any) -> float | None:
    if x is None:
        return None
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def round_metric(value: float | None, metric: str) -> float | None:
    if value is None:
        return None
    return round(value, ROUND_DECIMALS.get(metric, 4))


def norm_structure(dataset: str, name: str) -> str:
    key = name.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if dataset == "ACDC":
        mapping = {
            "rv": "right ventricle",
            "right ventricle": "right ventricle",
            "right heart ventricle": "right ventricle",
            "myo": "myocardium",
            "myocardium": "myocardium",
            "left ventricular myocardium": "myocardium",
            "lv": "left ventricle",
            "left ventricle": "left ventricle",
            "left heart ventricle": "left ventricle",
            "left ventricular cavity": "left ventricle",
        }
        return mapping.get(key, key)
    if dataset == "BTCV_cervix":
        mapping = {
            "class 1": "bladder",
            "class_1": "bladder",
            "bladder": "bladder",
            "urinary bladder": "bladder",
            "class 2": "uterus",
            "class_2": "uterus",
            "uterus": "uterus",
            "class 3": "rectum",
            "class_3": "rectum",
            "rectum": "rectum",
            "class 4": "small bowel",
            "class_4": "small bowel",
            "small bowel": "small bowel",
        }
        return mapping.get(name.strip().lower(), mapping.get(key, key))
    if dataset == "PROMISE12":
        return "prostate"
    return key


def metric_mean_std_from_flat(obj: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    return clean_number(obj.get(f"{metric}_mean")), clean_number(obj.get(f"{metric}_std"))


def metric_mean_std(obj: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    if f"{metric}_mean" in obj or f"{metric}_std" in obj:
        return metric_mean_std_from_flat(obj, metric)
    nested = obj.get(metric)
    if isinstance(nested, dict):
        return clean_number(nested.get("mean")), clean_number(nested.get("std"))
    return clean_number(obj.get(metric)), clean_number(obj.get(f"{metric}_std"))


def load_result(method: str, dataset: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    n_cases = data.get("n_cases")
    if n_cases is None and isinstance(data.get("per_patient"), dict):
        n_cases = len(data["per_patient"])
    if n_cases is None and isinstance(data.get("per_sample"), list):
        n_cases = len({row.get("patient_frame") for row in data["per_sample"] if isinstance(row, dict) and row.get("patient_frame")})
    if n_cases is None:
        n_cases = data.get("overall", {}).get("total_samples")

    overall_src = data.get("overall", {})
    overall_row = {
        "method": method,
        "dataset": dataset,
        "n_cases": n_cases,
        "source_file": str(path),
    }
    long_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        mean, std = metric_mean_std(overall_src, metric)
        mean = round_metric(mean, metric)
        std = round_metric(std, metric)
        overall_row[f"{metric}_mean"] = mean
        overall_row[f"{metric}_std"] = std
        long_rows.append({
            "method": method,
            "dataset": dataset,
            "scope": "overall",
            "structure": "overall",
            "metric": "dsc" if metric == "dice" else metric,
            "mean": mean,
            "std": std,
            "n_cases": n_cases,
            "source_file": str(path),
        })

    per_src = data.get("per_class") or data.get("per_label") or {}
    per_rows: list[dict[str, Any]] = []
    for raw_name, metrics in per_src.items():
        if not isinstance(metrics, dict):
            continue
        structure = norm_structure(dataset, str(raw_name))
        row = {
            "method": method,
            "dataset": dataset,
            "structure": structure,
            "source_structure": raw_name,
            "n_cases": metrics.get("count") or metrics.get("num_samples") or n_cases,
            "source_file": str(path),
        }
        for metric in METRICS:
            mean, std = metric_mean_std(metrics, metric)
            mean = round_metric(mean, metric)
            std = round_metric(std, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            long_rows.append({
                "method": method,
                "dataset": dataset,
                "scope": "per_structure",
                "structure": structure,
                "source_structure": raw_name,
                "metric": "dsc" if metric == "dice" else metric,
                "mean": mean,
                "std": std,
                "n_cases": row["n_cases"],
                "source_file": str(path),
            })
        per_rows.append(row)
    return overall_row, per_rows, long_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tex_method(method: str) -> str:
    return "BiomedParse-v2" if method == "BiomedParse v2" else method


def tex_num(value: float | None, metric: str) -> str:
    if value is None:
        return "--"
    if metric in {"iou", "dsc", "nsd"}:
        return f"{value * 100:.2f}"
    return f"{value:.2f}"


def tex_metric_cell(row: dict[str, Any], metric: str, mark: str = "", gray: bool = False) -> str:
    mean = tex_num(row.get(f"{metric}_mean"), metric)
    std = tex_num(row.get(f"{metric}_std"), metric)
    cell = f"{mean} $\\pm$ {std}"
    if mark == "bold":
        cell = f"\\textbf{{{cell}}}"
    elif mark == "underline":
        cell = f"\\underline{{{cell}}}"
    if gray:
        cell = f"\\cellcolor{{gray!20}}{cell}"
    return cell


def is_leakage(dataset: str, method: str) -> bool:
    return method in LEAKAGE_METHODS_BY_DATASET.get(dataset, set())


def valid_rank_marks(dataset: str, rows: list[dict[str, Any]], metric: str) -> dict[str, str]:
    candidates = [
        row for row in rows
        if row["method"] != "SAM3-Full" and not is_leakage(dataset, row["method"]) and row.get(f"{metric}_mean") is not None
    ]
    reverse = metric != "hd95"
    candidates.sort(key=lambda row: row[f"{metric}_mean"], reverse=reverse)
    marks = {}
    if candidates:
        marks[candidates[0]["method"]] = "bold"
    if len(candidates) > 1:
        marks[candidates[1]["method"]] = "underline"
    return marks


def render_main_tables(rows: list[dict[str, Any]]) -> str:
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["method"]] = row

    out = ["% Generated by gq_paper/cmpb/aggregate_cmpb_results.py.", ""]
    for dataset in sorted(by_dataset, key=lambda d: DATASET_ORDER.get(d, 99)):
        dataset_rows = list(by_dataset[dataset].values())
        table_metrics = ["dsc", "hd95", "iou", "nsd"]
        marks = {metric: valid_rank_marks(dataset, dataset_rows, metric) for metric in table_metrics}
        out.extend([
            "\\begin{table}[!htbp]",
            "\\centering",
            f"\\caption{{Quantitative comparison on {DATASET_LABELS[dataset]}. Values are reported as mean $\\pm$ standard deviation. Gray cells indicate data-leakage cases. Best valid scores, excluding leakage cases and the SAM3-Full upper bound, are in bold; second-best scores are underlined.}}",
            f"\\label{{tab:main_results_{DATASET_TABLE_LABELS[dataset]}}}",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l c c c c c}",
            "\\toprule",
            "\\textbf{Method} & \\textbf{Sup.} & \\textbf{DSC (\\%)} $\\uparrow$ & \\textbf{HD95} $\\downarrow$ & \\textbf{IoU (\\%)} $\\uparrow$ & \\textbf{NSD (\\%)} $\\uparrow$ \\\\",
            "\\midrule",
            "\\textit{Text-prompted models} & & & & & \\\\",
        ])
        for method in TEXT_PROMPTED_METHODS:
            row = by_dataset[dataset][method]
            gray = is_leakage(dataset, method)
            out.append(
                f"{tex_method(method)} & Full & "
                f"{tex_metric_cell(row, 'dsc', marks['dsc'].get(method, ''), gray)} & "
                f"{tex_metric_cell(row, 'hd95', marks['hd95'].get(method, ''), gray)} & "
                f"{tex_metric_cell(row, 'iou', marks['iou'].get(method, ''), gray)} & "
                f"{tex_metric_cell(row, 'nsd', marks['nsd'].get(method, ''), gray)} \\\\"
            )
        out.extend(["\\midrule", "\\textit{Scribble-supervised static models} & & & & & \\\\"])
        for method in STATIC_SCRIBBLE_METHODS:
            row = by_dataset[dataset][method]
            out.append(
                f"{tex_method(method)} & Scrib. & "
                f"{tex_metric_cell(row, 'dsc', marks['dsc'].get(method, ''))} & "
                f"{tex_metric_cell(row, 'hd95', marks['hd95'].get(method, ''))} & "
                f"{tex_metric_cell(row, 'iou', marks['iou'].get(method, ''))} & "
                f"{tex_metric_cell(row, 'nsd', marks['nsd'].get(method, ''))} \\\\"
            )
        out.extend(["\\midrule", "\\textit{Ours} & & & & & \\\\"])
        for method in OURS_METHODS:
            row = by_dataset[dataset][method]
            sup = "\\textbf{Scrib.}" if method == "SAM3-Scribble" else "Full"
            name = "\\textbf{SAM3-Scribble}" if method == "SAM3-Scribble" else "SAM3-Full"
            out.append(
                f"{name} & {sup} & "
                f"{tex_metric_cell(row, 'dsc', marks['dsc'].get(method, ''))} & "
                f"{tex_metric_cell(row, 'hd95', marks['hd95'].get(method, ''))} & "
                f"{tex_metric_cell(row, 'iou', marks['iou'].get(method, ''))} & "
                f"{tex_metric_cell(row, 'nsd', marks['nsd'].get(method, ''))} \\\\"
            )
        out.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ])
    return "\n".join(out)


def render_supplement_tables(rows: list[dict[str, Any]]) -> str:
    by_dataset_structure = {}
    for row in rows:
        by_dataset_structure.setdefault(row["dataset"], {}).setdefault(row["structure"], {})[row["method"]] = row

    out = [
        "\\section*{Supplementary per-structure results}",
        "Tables~\\ref{tab:supp_per_structure_acdc}--\\ref{tab:supp_per_structure_promise12} report per-structure results for each dataset. DSC, IoU, and NSD are shown as percentages; HD95 is reported in millimeters.",
        "",
    ]
    for dataset in sorted(by_dataset_structure, key=lambda d: DATASET_ORDER.get(d, 99)):
        out.extend([
            "\\begin{table}[!p]",
            "\\centering",
            f"\\caption{{Per-structure quantitative results on {DATASET_LABELS[dataset]}. Values are reported as mean $\\pm$ standard deviation.}}",
            f"\\label{{tab:supp_per_structure_{DATASET_TABLE_LABELS[dataset]}}}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l l c c c c}",
            "\\toprule",
            "\\textbf{Structure} & \\textbf{Method} & \\textbf{DSC (\\%)} $\\uparrow$ & \\textbf{HD95} $\\downarrow$ & \\textbf{IoU (\\%)} $\\uparrow$ & \\textbf{NSD (\\%)} $\\uparrow$ \\\\",
            "\\midrule",
        ])
        structures = STRUCTURE_ORDER.get(dataset, sorted(by_dataset_structure[dataset]))
        for structure in structures:
            first = True
            methods = TEXT_PROMPTED_METHODS + STATIC_SCRIBBLE_METHODS + OURS_METHODS
            for method in methods:
                row = by_dataset_structure[dataset][structure].get(method)
                if row is None:
                    continue
                gray = is_leakage(dataset, method)
                structure_label = structure if first else ""
                first = False
                out.append(
                    f"{structure_label} & {tex_method(method)} & "
                    f"{tex_metric_cell(row, 'dsc', gray=gray)} & "
                    f"{tex_metric_cell(row, 'hd95', gray=gray)} & "
                    f"{tex_metric_cell(row, 'iou', gray=gray)} & "
                    f"{tex_metric_cell(row, 'nsd', gray=gray)} \\\\"
                )
        out.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ])
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    missing = []
    overall_rows = []
    per_rows = []
    long_rows = []

    for method, dataset, path_str in SOURCES:
        path = Path(path_str)
        entry = {"method": method, "dataset": dataset, "source_file": str(path), "exists": path.is_file()}
        manifest.append(entry)
        if not path.is_file():
            missing.append(entry)
            continue
        overall, per, long = load_result(method, dataset, path)
        overall_rows.append(overall)
        per_rows.extend(per)
        long_rows.extend(long)

    def sort_key(row: dict[str, Any]):
        return (
            DATASET_ORDER.get(row.get("dataset"), 99),
            METHOD_ORDER.get(row.get("method"), 99),
            row.get("structure", ""),
            row.get("metric", ""),
        )

    overall_rows.sort(key=sort_key)
    per_rows.sort(key=sort_key)
    long_rows.sort(key=sort_key)

    metric_cols = []
    for metric in METRICS:
        label = "dsc" if metric == "dice" else metric
        metric_cols.extend([f"{label}_mean", f"{label}_std"])

    # Rename dice columns to dsc in CSV-facing rows.
    def rename_dice(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if "dice_mean" in out:
            out["dsc_mean"] = out.pop("dice_mean")
        if "dice_std" in out:
            out["dsc_std"] = out.pop("dice_std")
        return out

    overall_csv_rows = [rename_dice(r) for r in overall_rows]
    per_csv_rows = [rename_dice(r) for r in per_rows]

    write_csv(
        OUT_DIR / "overall_metrics_wide.csv",
        overall_csv_rows,
        ["dataset", "method", "n_cases", "iou_mean", "iou_std", "dsc_mean", "dsc_std", "hd95_mean", "hd95_std", "nsd_mean", "nsd_std", "source_file"],
    )
    write_csv(
        OUT_DIR / "per_structure_metrics_wide.csv",
        per_csv_rows,
        ["dataset", "method", "structure", "source_structure", "n_cases", "iou_mean", "iou_std", "dsc_mean", "dsc_std", "hd95_mean", "hd95_std", "nsd_mean", "nsd_std", "source_file"],
    )
    write_csv(
        OUT_DIR / "metrics_long.csv",
        long_rows,
        ["dataset", "method", "scope", "structure", "source_structure", "metric", "mean", "std", "n_cases", "source_file"],
    )

    (OUT_DIR / "source_manifest.json").write_text(json.dumps({"sources": manifest, "missing": missing}, indent=2), encoding="utf-8")
    (OUT_DIR / "collected_results.json").write_text(
        json.dumps(
            {
                "overall": overall_rows,
                "per_structure": per_rows,
                "long": long_rows,
                "missing_sources": missing,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "generated_main_tables.tex").write_text(render_main_tables(overall_csv_rows), encoding="utf-8")
    (OUT_DIR / "generated_supplement_tables.tex").write_text(render_supplement_tables(per_csv_rows), encoding="utf-8")
    print(f"Wrote {len(overall_rows)} overall rows, {len(per_rows)} per-structure rows, {len(long_rows)} long rows to {OUT_DIR}")
    if missing:
        print("Missing sources:")
        for item in missing:
            print(f"  {item['method']} {item['dataset']}: {item['source_file']}")


if __name__ == "__main__":
    main()
