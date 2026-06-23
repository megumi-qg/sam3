#!/usr/bin/env python3
"""Plot per-case DSC distributions for the CMPB manuscript."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/home/gaoqi/sam3/gq_paper/cmpb/elsarticle/per_case_dsc_distribution.png")

SOURCES = {
    ("ScribFormer", "ACDC"): "/home/gaoqi/ScribFormer/model/ACDC_CMPB/scribformer_cmpb/scribble/test_eval_summary.json",
    ("ScribbleBench", "ACDC"): "/home/gaoqi/dataset/nnU-Net/results/Dataset120_ACDCS_CMPB/test_eval_summary_sam3_metrics.json",
    ("EFFDNet", "ACDC"): "/home/gaoqi/EFFDNet/model/ACDC_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json",
    ("SAM3-Scribble", "ACDC"): "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_acdc_0.7/evaluation_results_acdc.json",
    ("SAM3-Full", "ACDC"): "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_acdc_0.7/evaluation_results_acdc.json",
    ("ScribFormer", "BTCV_cervix"): "/home/gaoqi/ScribFormer/model/BTCV_CMPB/scribformer_cmpb/scribble/test_eval_summary.json",
    ("ScribbleBench", "BTCV_cervix"): "/home/gaoqi/dataset/nnU-Net/results/Dataset121_BTCVS_CMPB/test_eval_summary_sam3_metrics.json",
    ("EFFDNet", "BTCV_cervix"): "/home/gaoqi/EFFDNet/model/BTCV_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json",
    ("SAM3-Scribble", "BTCV_cervix"): "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json",
    ("SAM3-Full", "BTCV_cervix"): "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json",
    ("ScribFormer", "PROMISE12"): "/home/gaoqi/ScribFormer/model/PROMISE12_CMPB/scribformer_cmpb/scribble/test_eval_summary.json",
    ("ScribbleBench", "PROMISE12"): "/home/gaoqi/dataset/nnU-Net/results/Dataset122_PROMISES_CMPB/test_eval_summary_sam3_metrics.json",
    ("EFFDNet", "PROMISE12"): "/home/gaoqi/EFFDNet/model/PROMISE12_CMPB/EFFDNet_cmpb/scribble/test_eval_summary.json",
    ("SAM3-Scribble", "PROMISE12"): "/home/gaoqi/sam3/gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json",
    ("SAM3-Full", "PROMISE12"): "/home/gaoqi/sam3/gq_experiment/cmpb/full_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json",
}

DATASETS = ["ACDC", "BTCV_cervix", "PROMISE12"]
METHODS = ["ScribFormer", "ScribbleBench", "EFFDNet", "SAM3-Scribble", "SAM3-Full"]
LABELS = ["ScribFormer", "ScribbleBench", "EFFDNet", "SAM3-\nScribble", "SAM3-\nFull"]
COLORS = ["#9ca3af", "#60a5fa", "#f59e0b", "#10b981", "#374151"]


def norm_case(case_id: str) -> str:
    case_id = re.sub(r"\.nii(\.gz)?$", "", str(case_id))
    case_id = re.sub(r"(?i)-image$", "", case_id)
    case_id = re.sub(r"(?i)^case", "", case_id)
    return case_id


def load_dice(path: str) -> list[float]:
    data = json.loads(Path(path).read_text())
    per_patient = data.get("per_patient", {})
    rows = []
    for case_id, record in sorted(per_patient.items(), key=lambda x: norm_case(x[0])):
        overall = record.get("overall", record)
        if "dice" in overall:
            rows.append(float(overall["dice"]) * 100.0)
    return rows


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        values = [load_dice(SOURCES[(method, dataset)]) for method in METHODS]
        positions = np.arange(1, len(METHODS) + 1)
        bp = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.3)
        for whisker in bp["whiskers"]:
            whisker.set_color("#6b7280")
        for cap in bp["caps"]:
            cap.set_color("#6b7280")

        for pos, vals, color in zip(positions, values, COLORS):
            jitter = rng.normal(0, 0.045, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=15, color=color, alpha=0.72, linewidths=0)

        title = "BTCV_cervix" if dataset == "BTCV_cervix" else dataset
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(LABELS, rotation=25, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_ylim(30, 100)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Per-case DSC (%)")
    fig.tight_layout(w_pad=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=350, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
