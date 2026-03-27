#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 scribble 标注中是否存在会导致训练报错的退化标注。

报错原因说明：
  RuntimeError: Input and output sizes should be greater than 0, but got input (H: 288, W: 288) output (H: 1, W: 0)

当 scribble 为「单像素宽」的垂直线（或「单像素高」的水平线）时：
- bbox 会退化为 width=0 或 height=0（例如 x_min==x_max 或 y_min==y_max）
- 若 mask 按 bbox 裁剪，会得到 (H, 0) 或 (0, W) 的无效尺寸
- 在 PartialMasks loss 的 interpolate 中会触发上述错误

本脚本检查：
1. bbox 宽或高 <= 0 的标注
2. 解码后 mask 空间尺寸为 0 的标注
3. 解码后 mask 有效像素数 <= 1 的极细 scribble（可能产生退化 bbox）

用法：
  python gq_scripts/tool/check_scribble_degenerate.py \
    --json_paths /path/to/acdc/train/scribble_annotations.coco.json \
                /path/to/btcv/train/scribble_annotations.coco.json \
                /path/to/promise12/train/scribble_annotations.coco.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util
from tqdm import tqdm


def decode_segmentation(segm, h: int, w: int) -> np.ndarray:
    """解码 segmentation（RLE 或 polygon）为二值 mask"""
    if segm is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(segm, list):
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm.get("counts", ""), list):
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        rle = segm
    mask = mask_util.decode(rle)
    if len(mask.shape) == 3:
        mask = (mask.sum(axis=2) > 0).astype(np.uint8)
    return mask


def check_annotation(ann: dict, im_info: dict) -> list:
    """
    检查单条 annotation 是否存在退化情况。
    返回问题描述列表，空列表表示无问题。
    """
    issues = []
    h, w = im_info["height"], im_info["width"]

    # 1. 检查 bbox（COCO 格式: [x, y, width, height]）
    if "bbox" in ann and ann["bbox"] is not None:
        x, y, bw, bh = ann["bbox"]
        if bw <= 0 or bh <= 0:
            issues.append(f"bbox 退化: [x={x}, y={y}, w={bw}, h={bh}] (宽或高<=0)")
        elif bw < 2 or bh < 2:
            issues.append(f"bbox 过小: [x={x}, y={y}, w={bw}, h={bh}] (可能为单像素线)")

    # 2. 从 segmentation 解码并检查 mask
    segm = ann.get("segmentation")
    if segm is None:
        return issues

    try:
        mask = decode_segmentation(segm, h, w)
    except Exception as e:
        issues.append(f"解码 segmentation 失败: {e}")
        return issues

    # 检查 mask 空间尺寸
    mh, mw = mask.shape
    if mh <= 0 or mw <= 0:
        issues.append(f"mask 尺寸无效: ({mh}, {mw})")

    # 检查有效像素数
    n_pos = int(np.sum(mask > 0))
    if n_pos == 0:
        issues.append("mask 全零（无有效 scribble）")
    elif n_pos == 1:
        issues.append("mask 仅 1 像素（单点 scribble，易产生退化 bbox）")

    # 3. 从 mask 计算 bbox，检查是否退化
    if n_pos > 0:
        rows, cols = np.where(mask > 0)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1
        if box_w <= 0 or box_h <= 0:
            issues.append(f"从 mask 计算的 bbox 退化: w={box_w}, h={box_h}")
        elif box_w == 1 or box_h == 1:
            issues.append(
                f"从 mask 计算的 bbox 为单像素线: w={box_w}, h={box_h}"
                " (训练时可能产生 H:1,W:0 或 H:0,W:1 的 interpolate 错误)"
            )

    return issues


def check_coco_json(json_path: Path) -> dict:
    """检查单个 COCO JSON 文件，返回统计和问题列表"""
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    total_anns = 0
    bad_anns = []
    bad_images = set()

    for ann in tqdm(coco["annotations"], desc=str(json_path.name), leave=False):
        total_anns += 1
        img_id = ann["image_id"]
        im_info = img_id_to_info.get(img_id)
        if im_info is None:
            bad_anns.append((ann, ["image_id 不存在于 images"]))
            continue

        issues = check_annotation(ann, im_info)
        if issues:
            bad_anns.append((ann, issues))
            bad_images.add(img_id)

    return {
        "path": str(json_path),
        "total_images": len(images),
        "total_annotations": total_anns,
        "bad_annotations": len(bad_anns),
        "bad_images": len(bad_images),
        "details": bad_anns,
        "bad_image_ids": list(bad_images),
    }


def main():
    parser = argparse.ArgumentParser(
        description="检查 scribble COCO JSON 中的退化标注"
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        type=str,
        help="scribble_annotations.coco.json 文件路径（可多个）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="将详细结果写入 JSON 文件",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="打印每条问题标注的详情",
    )
    args = parser.parse_args()

    all_results = []
    for path in args.json_paths:
        p = Path(path)
        if not p.exists():
            print(f"[跳过] 文件不存在: {p}")
            continue
        result = check_coco_json(p)
        all_results.append(result)

    # 汇总输出
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)

    total_bad = 0
    for r in all_results:
        name = Path(r["path"]).name
        print(f"\n【{name}】")
        print(f"  总标注数: {r['total_annotations']}")
        print(f"  问题标注数: {r['bad_annotations']}")
        print(f"  涉及图像数: {r['bad_images']}")
        total_bad += r["bad_annotations"]

        if args.verbose and r["details"]:
            print("  问题详情:")
            for ann, issues in r["details"][:20]:  # 最多显示 20 条
                print(f"    ann_id={ann.get('id')} image_id={ann.get('image_id')}:")
                for iss in issues:
                    print(f"      - {iss}")
            if len(r["details"]) > 20:
                print(f"    ... 还有 {len(r['details']) - 20} 条")

    print("\n" + "-" * 60)
    if total_bad == 0:
        print("未发现退化标注。若训练仍报错，可能是 transform 或 bbox 裁剪导致。")
    else:
        print(f"共发现 {total_bad} 条可能问题的标注。")
        print("建议：在预处理中过滤 bbox 宽或高<=1 的标注，或增加最小有效像素数阈值。")

    if args.output:
        out_path = Path(args.output)
        # 将 details 中的 ann 转为可序列化（numpy 等转成 list）
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(x) for x in obj]
            return obj

        out_data = []
        for r in all_results:
            out_data.append({
                "path": r["path"],
                "total_annotations": r["total_annotations"],
                "bad_annotations": r["bad_annotations"],
                "bad_images": r["bad_images"],
                "bad_image_ids": r["bad_image_ids"],
                "details": [
                    {"ann_id": a.get("id"), "image_id": a.get("image_id"), "issues": i}
                    for a, i in r["details"]
                ],
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已写入: {out_path}")


if __name__ == "__main__":
    main()
