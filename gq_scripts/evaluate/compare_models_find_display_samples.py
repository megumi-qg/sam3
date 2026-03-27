#!/usr/bin/env python3
"""
从 Ours / SAT-Pro / scribble_bench / EFFDNet 四个模型的评估结果中，
筛选「我们的模型在该样本上 overall 及各个解剖区域均优于三个对比模型」的样本，
并按领先幅度排序，便于论文插图选样。

支持 BTCV_cervix 和 PROMISE12 数据集；可通过 --dataset 参数选择数据集。

用法:
  python gq_scripts/evaluate/compare_models_find_display_samples.py --dataset btcv
  python gq_scripts/evaluate/compare_models_find_display_samples.py --dataset promise12
  # 或指定路径
  python compare_models_find_display_samples.py --dataset btcv --ours ... --sat ... --scribble ... --effdnet ...
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

# 数据集配置
DATASET_CONFIGS = {
    "btcv": {
        "default_paths": {
            "ours": "/home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_btcv_0.7/evaluation_results_btcv.json",
            "sat": "/home/gaoqi/SAT/gq_dataset/BTCV/test/results_pro/results.json",
            "scribble": "/home/gaoqi/dataset/nnU-Net/results/Dataset021_BTCVS/evaluation_results.json",
            "effdnet": "/home/gaoqi/EFFDNet/model/BTCV_cervix/EFFDNet_fold1/scribble/test_eval_summary.json",
        },
        "regions": ["bladder", "rectum", "small_bowel", "uterus"],
        "ours_class_to_region": {
            "bladder": "bladder",
            "rectum": "rectum",
            "small bowel": "small_bowel",
            "uterus": "uterus",
        },
        "sat_label_to_region": {
            "urinary bladder": "bladder",
            "uterus": "uterus",
            "rectum": "rectum",
            "small bowel": "small_bowel",
        },
        "scribble_region_map": {
            "bladder": "bladder",
            "uterus": "uterus",
            "rectum": "rectum",
            "small_bowel": "small_bowel",
        },
        "effdnet_class_to_region": {
            "class_1": "bladder",
            "class_2": "uterus",
            "class_3": "rectum",
            "class_4": "small_bowel",
        },
        "region_names": {
            "bladder": "膀胱 (bladder)",
            "rectum": "直肠 (rectum)",
            "small_bowel": "小肠 (small_bowel)",
            "uterus": "子宫 (uterus)",
        },
        "sat_pid_normalize": lambda pid: pid.replace("-Image", "") if isinstance(pid, str) else pid,
        "scribble_pid_normalize": lambda pid: pid,
        "effdnet_pid_normalize": lambda pid: (
            pid.replace("-Image.nii.gz", "") if pid.endswith("-Image.nii.gz")
            else pid.replace(".nii.gz", "") if pid.endswith(".nii.gz")
            else pid
        ),
    },
    "promise12": {
        "default_paths": {
            "ours": "/home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/evaluation_results_promise12.json",
            "sat": "/home/gaoqi/SAT/gq_dataset/PROMISE12/test/results_pro/results.json",
            "scribble": "/home/gaoqi/dataset/nnU-Net/results/Dataset022_PROMISES/evaluation_results.json",
            "effdnet": "/home/gaoqi/EFFDNet/model/PROMISE12/EFFDNet_fold1/scribble/test_eval_summary.json",
        },
        "regions": ["prostate"],
        "ours_class_to_region": {
            "prostate": "prostate",
        },
        "sat_label_to_region": {
            "prostate": "prostate",
        },
        "scribble_region_map": {
            "prostate": "prostate",
        },
        "effdnet_class_to_region": {
            "Prostate": "prostate",
        },
        "region_names": {
            "prostate": "前列腺 (prostate)",
        },
        "sat_pid_normalize": lambda pid: pid.replace("Case", "") if isinstance(pid, str) and pid.startswith("Case") else pid,
        "scribble_pid_normalize": lambda pid: pid.replace("Case", "") if isinstance(pid, str) and pid.startswith("Case") else pid,
        "effdnet_pid_normalize": lambda pid: (
            pid.replace("Case", "").replace(".nii.gz", "") if isinstance(pid, str) and pid.startswith("Case")
            else pid.replace(".nii.gz", "") if pid.endswith(".nii.gz")
            else pid
        ),
    },
}

# 全局变量，将在 main 中根据数据集设置
REGIONS = []
OURS_CLASS_TO_REGION = {}
SAT_LABEL_TO_REGION = {}
SCRIBBLE_REGION_MAP = {}
EFFDNET_CLASS_TO_REGION = {}
REGION_NAMES = {}


def load_ours(data: dict):
    """Ours: per_patient[pid].overall.dice, per_patient[pid].per_class[class].dice"""
    overall = {}
    per_region = defaultdict(dict)
    for pid, p in data.get("per_patient", {}).items():
        overall[pid] = p["overall"]["dice"]
        for cls, region in OURS_CLASS_TO_REGION.items():
            if cls in p.get("per_class", {}):
                per_region[pid][region] = p["per_class"][cls]["dice"]
    return overall, dict(per_region)


def load_sat(data: dict, sat_pid_normalize_fn):
    """SAT-Pro: per_sample list, group by patient_frame, aggregate by label -> region"""
    by_patient = defaultdict(lambda: {"overall_dice": [], **{r: [] for r in REGIONS}})
    for s in data.get("per_sample", []):
        raw_pid = s["patient_frame"]
        pid = sat_pid_normalize_fn(raw_pid)
        label = s["label"]
        dice = s["dice"]
        region = SAT_LABEL_TO_REGION.get(label)
        if region is not None:
            by_patient[pid][region].append(dice)
            by_patient[pid]["overall_dice"].append(dice)
    overall = {}
    per_region = defaultdict(dict)
    for pid, v in by_patient.items():
        if v["overall_dice"]:
            overall[pid] = sum(v["overall_dice"]) / len(v["overall_dice"])
        for r in REGIONS:
            if v[r]:
                per_region[pid][r] = sum(v[r]) / len(v[r])
    return overall, dict(per_region)


def load_scribble(data: dict, scribble_pid_normalize_fn):
    """scribble_bench: per_patient_per_region[pid][region].Dice (capital D)"""
    overall = {}
    per_region = defaultdict(dict)
    for raw_pid, regions in data.get("per_patient_per_region", {}).items():
        pid = scribble_pid_normalize_fn(raw_pid)
        dices = []
        for key, region in SCRIBBLE_REGION_MAP.items():
            if key in regions and "Dice" in regions[key]:
                d = regions[key]["Dice"]
                per_region[pid][region] = d
                dices.append(d)
        if dices:
            overall[pid] = sum(dices) / len(dices)
    return overall, dict(per_region)


def load_effdnet(data: dict, effdnet_pid_normalize_fn):
    """EFFDNet: per_patient[pid].overall.dice, per_class"""
    overall = {}
    per_region = defaultdict(dict)
    for key, p in data.get("per_patient", {}).items():
        pid = effdnet_pid_normalize_fn(key)
        overall[pid] = p["overall"]["dice"]
        for cls, region in EFFDNET_CLASS_TO_REGION.items():
            if cls in p.get("per_class", {}):
                per_region[pid][region] = p["per_class"][cls]["dice"]
    return overall, dict(per_region)


def main():
    parser = argparse.ArgumentParser(description="Compare models and find display samples.")
    parser.add_argument(
        "--dataset",
        choices=["btcv", "promise12"],
        default="btcv",
        help="数据集类型: btcv 或 promise12 (默认: btcv)"
    )
    parser.add_argument("--ours", default=None, help="Ours evaluation JSON (默认使用数据集配置)")
    parser.add_argument("--sat", default=None, help="SAT-Pro results JSON (默认使用数据集配置)")
    parser.add_argument("--scribble", default=None, help="scribble_bench evaluation JSON (默认使用数据集配置)")
    parser.add_argument("--effdnet", default=None, help="EFFDNet test_eval_summary JSON (默认使用数据集配置)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path for results (optional)")
    args = parser.parse_args()

    # 根据数据集选择配置
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {args.dataset}，支持的数据集: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[args.dataset]
    
    # 设置全局变量
    global REGIONS, OURS_CLASS_TO_REGION, SAT_LABEL_TO_REGION, SCRIBBLE_REGION_MAP, EFFDNET_CLASS_TO_REGION, REGION_NAMES
    REGIONS = config["regions"]
    OURS_CLASS_TO_REGION = config["ours_class_to_region"]
    SAT_LABEL_TO_REGION = config["sat_label_to_region"]
    SCRIBBLE_REGION_MAP = config["scribble_region_map"]
    EFFDNET_CLASS_TO_REGION = config["effdnet_class_to_region"]
    REGION_NAMES = config["region_names"]
    
    # 获取路径（优先使用命令行参数，否则使用数据集默认路径）
    default_paths = config["default_paths"]
    ours_path = Path(args.ours) if args.ours else Path(default_paths["ours"])
    sat_path = Path(args.sat) if args.sat else Path(default_paths["sat"])
    scribble_path = Path(args.scribble) if args.scribble else Path(default_paths["scribble"])
    effdnet_path = Path(args.effdnet) if args.effdnet else Path(default_paths["effdnet"])

    for p, name in [(ours_path, "Ours"), (sat_path, "SAT"), (scribble_path, "Scribble"), (effdnet_path, "EFFDNet")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} 结果文件不存在: {p}")

    with open(ours_path) as f:
        ours_data = json.load(f)
    with open(sat_path) as f:
        sat_data = json.load(f)
    with open(scribble_path) as f:
        scribble_data = json.load(f)
    with open(effdnet_path) as f:
        effdnet_data = json.load(f)

    # 获取 patient id 归一化函数
    sat_pid_normalize_fn = config["sat_pid_normalize"]
    scribble_pid_normalize_fn = config["scribble_pid_normalize"]
    effdnet_pid_normalize_fn = config["effdnet_pid_normalize"]

    o_overall, o_region = load_ours(ours_data)
    s_overall, s_region = load_sat(sat_data, sat_pid_normalize_fn)
    b_overall, b_region = load_scribble(scribble_data, scribble_pid_normalize_fn)
    e_overall, e_region = load_effdnet(effdnet_data, effdnet_pid_normalize_fn)

    common = set(o_overall) & set(s_overall) & set(b_overall) & set(e_overall)
    print(f"四个模型共有样本数: {len(common)}")

    # 每个样本在 overall 上是否优于三者
    def ours_wins_overall(pid):
        o, s, b, e = o_overall[pid], s_overall[pid], b_overall[pid], e_overall[pid]
        return o > s and o > b and o > e

    def ours_wins_region(pid, region):
        o = o_region.get(pid, {}).get(region)
        s = s_region.get(pid, {}).get(region)
        b = b_region.get(pid, {}).get(region)
        e = e_region.get(pid, {}).get(region)
        if o is None or s is None or b is None or e is None:
            return False, None
        return (o > s and o > b and o > e), o

    # Overall 候选 + 领先幅度
    overall_candidates = [pid for pid in common if ours_wins_overall(pid)]
    overall_margins = []
    for pid in overall_candidates:
        o, s, b, e = o_overall[pid], s_overall[pid], b_overall[pid], e_overall[pid]
        margin_min = o - min(s, b, e)
        margin_avg = (o - s) + (o - b) + (o - e)
        overall_margins.append((pid, o, s, b, e, margin_min, margin_avg))
    overall_margins.sort(key=lambda x: (-x[5], -x[6]))  # 按 margin_min 降序

    # 各区域候选 + 领先幅度
    region_candidates = {r: [] for r in REGIONS}
    region_margins = {r: [] for r in REGIONS}
    for pid in common:
        for r in REGIONS:
            win, o_val = ours_wins_region(pid, r)
            if not win or o_val is None:
                continue
            s = s_region.get(pid, {}).get(r)
            b = b_region.get(pid, {}).get(r)
            e = e_region.get(pid, {}).get(r)
            if s is None or b is None or e is None:
                continue
            margin_min = o_val - min(s, b, e)
            margin_avg = (o_val - s) + (o_val - b) + (o_val - e)
            region_candidates[r].append(pid)
            region_margins[r].append((pid, o_val, s, b, e, margin_min, margin_avg))
    for r in REGIONS:
        region_margins[r].sort(key=lambda x: (-x[5], -x[6]))

    # 同时满足 overall 与各区域都赢的样本（用于插图：整体和每个结构都更好）
    all_win_pids = [
        pid for pid in overall_candidates
        if all(pid in region_candidates[r] for r in REGIONS)
    ]
    all_win_margins = []
    for pid in all_win_pids:
        o, s, b, e = o_overall[pid], s_overall[pid], b_overall[pid], e_overall[pid]
        margin_min = o - min(s, b, e)
        all_win_margins.append((pid, o, s, b, e, margin_min))
    all_win_margins.sort(key=lambda x: -x[5])

    # ----- 打印 -----
    print("\n" + "=" * 60)
    print("一、Overall：我们优于 SAT / scribble_bench / EFFDNet 的样本数")
    print("=" * 60)
    print(f"候选数: {len(overall_candidates)} / {len(common)}")
    print("\n按「最小领先幅度」排序 (ours - min(sat, scribble, effdnet))，前 20 个:")
    print(f"{'patient_id':<25} {'Ours':>8} {'SAT':>8} {'Scribble':>10} {'EFFDNet':>10} {'margin_min':>10}")
    for t in overall_margins[:20]:
        print(f"{t[0]:<25} {t[1]:.4f}   {t[2]:.4f}   {t[3]:.4f}     {t[4]:.4f}     {t[5]:.4f}")

    print("\n" + "=" * 60)
    print("二、各解剖区域：我们优于三者的样本数及排序（前 10）")
    print("=" * 60)
    for r in REGIONS:
        print(f"\n--- {REGION_NAMES[r]} ---")
        print(f"候选数: {len(region_candidates[r])} / {len(common)}")
        print(f"{'patient_id':<25} {'Ours':>8} {'SAT':>8} {'Scribble':>10} {'EFFDNet':>10} {'margin_min':>10}")
        for t in region_margins[r][:10]:
            print(f"{t[0]:<25} {t[1]:.4f}   {t[2]:.4f}   {t[3]:.4f}     {t[4]:.4f}     {t[5]:.4f}")

    regions_str = "/".join(REGIONS)
    print("\n" + "=" * 60)
    print(f"三、Overall 与 {regions_str} 均优于三者的样本（最适合做插图）")
    print("=" * 60)
    print(f"候选数: {len(all_win_pids)}")
    if all_win_margins:
        print(f"\n按 overall 领先幅度排序:")
        print(f"{'patient_id':<25} {'Ours':>8} {'SAT':>8} {'Scribble':>10} {'EFFDNet':>10} {'margin_min':>10}")
        for t in all_win_margins[:25]:
            print(f"{t[0]:<25} {t[1]:.4f}   {t[2]:.4f}   {t[3]:.4f}     {t[4]:.4f}     {t[5]:.4f}")
    else:
        num_regions = len(REGIONS)
        print(f"无同时满足 overall 与 {num_regions} 个区域都赢的样本。可从上方的 overall 或单区域候选中选图。")

    # 输出 JSON 供后续画图/表格使用
    out = {
        "common_ids": sorted(common),
        "overall": {
            "candidates": overall_candidates,
            "ranked_by_margin_min": [
                {"patient_id": t[0], "ours": t[1], "sat": t[2], "scribble": t[3], "effdnet": t[4], "margin_min": t[5], "margin_avg": t[6]}
                for t in overall_margins
            ],
        },
        "per_region": {
            r: {
                "candidates": region_candidates[r],
                "ranked_by_margin_min": [
                    {"patient_id": t[0], "ours": t[1], "sat": t[2], "scribble": t[3], "effdnet": t[4], "margin_min": t[5], "margin_avg": t[6]}
                    for t in region_margins[r]
                ],
            }
            for r in REGIONS
        },
        "all_win_overall_and_regions": {
            "candidates": all_win_pids,
            "ranked": [
                {"patient_id": t[0], "ours": t[1], "sat": t[2], "scribble": t[3], "effdnet": t[4], "margin_min": t[5]}
                for t in all_win_margins
            ],
        },
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {out_path}")
    else:
        default_out = Path(ours_path).parent / "compare_models_display_samples.json"
        with open(default_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入默认路径: {default_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
