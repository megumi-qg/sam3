"""
SAM3 批量推理（2D 切片 → 按 volume 聚合，输出 pkl / 可选 PNG / NIfTI）

流程简述
--------
1. 读取测试目录下的 COCO JSON（默认 image_annotations.coco.json）与 images/。
2. 按病例与帧（或 BTCV/Promise12 的 default 帧）将切片分组为 volume。
3. 对每个 volume、每个类别、每张切片用 Sam3Processor 做文本提示分割；默认取置信度最高的一条 mask。
4. 将所有 volume 的结果写入 predictions.pkl，供 batch_evaluate.py 离线算指标。

LoRA checkpoint
---------------
训练时若在部分模块注入 LoRA、并对 mask_decoder / dot_prod_scoring 全量微调，则推理时必须用**相同**
的 lora_target_components 与 lora_unfreeze_components 构建模型，否则 state_dict 与结构不匹配。
本脚本默认与 ``sam3/train/configs/acdc/scribble_lora.yaml`` 一致：
LoRA → 5 个 encoder/decoder 组件；全量 → mask_decoder、dot_prod_scoring。
可通过 ``--use_lora true|false`` 覆盖自动检测（默认按权重中是否含 ``lora_A`` 判断）。

示例（ACDC 测试集 + scribble_tmi LoRA 最佳分割权重）::

    CUDA_VISIBLE_DEVICES=0 python gq_scripts/evaluate/batch_inference.py \
        --test_dir /home/gaoqi/dataset/using/acdc/processed/png_coco_sam3_fullframes_weak/test \
        --checkpoint_path /home/gaoqi/sam3/gq_experiment/acdc/scribble_tmi_lora/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
        --output_dir /home/gaoqi/sam3/gq_experiment/acdc/scribble_tmi_lora/inference_test \
        --confidence_threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

import sam3
from sam3 import build_sam3_image_model
from sam3.model.lora import merge_lora_into_sam3
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 与 sam3/train/configs/acdc/scribble_lora.yaml 中 LoRA 设置一致
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16.0
DEFAULT_LORA_TARGET_COMPONENTS = [
    "vision_encoder",
    "text_encoder",
    "geometry_encoder",
    "detr_encoder",
    "detr_decoder",
]
DEFAULT_LORA_UNFREEZE_COMPONENTS = [
    "mask_decoder",
    "dot_prod_scoring",
]


def extract_patient_frame(
    img_file: str,
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """从文件名解析 patient_id、frame_id、slice_idx、view（若适用）。"""
    basename = os.path.basename(img_file)

    match_acdc = re.match(r"patient(\d+)_frame(\d+)_slice(\d+)\.png", basename)
    if match_acdc:
        return (
            match_acdc.group(1),
            match_acdc.group(2),
            int(match_acdc.group(3)),
            None,
        )

    match_mms2 = re.match(r"(\d+)_(SA|LA)_(ED|ES)_slice(\d+)\.png", basename)
    if match_mms2:
        patient_id = match_mms2.group(1)
        view = match_mms2.group(2)
        phase = match_mms2.group(3)
        slice_idx = int(match_mms2.group(4))
        return patient_id, f"{view}_{phase}", slice_idx, view

    match_camus = re.match(
        r"patient(\d+)_([A-Z0-9]+)_(ED|ES)(?:_slice(\d+))?\.png", basename
    )
    if match_camus:
        patient_id = match_camus.group(1)
        view = match_camus.group(2)
        phase = match_camus.group(3)
        slice_idx_str = match_camus.group(4)
        slice_idx = int(slice_idx_str) if slice_idx_str else 0
        return patient_id, f"{view}_{phase}", slice_idx, view

    match_btcv = re.match(r"([\d]+)-Image_slice(\d+)\.png", basename)
    if match_btcv:
        return match_btcv.group(1), "default", int(match_btcv.group(2)), None

    match_promise12 = re.match(r"Case(\d+)_slice(\d+)\.png", basename)
    if match_promise12:
        return match_promise12.group(1), "default", int(match_promise12.group(2)), None

    match_isbi = re.match(r"patient(\d+)_slice(\d+)\.png", basename)
    if match_isbi:
        return match_isbi.group(1), "default", int(match_isbi.group(2)), None

    match_mscmr = re.match(r"subject(\d+)_([A-Za-z0-9]+)_slice(\d+)\.png", basename)
    if match_mscmr:
        patient_id = match_mscmr.group(1)
        phase = match_mscmr.group(2)
        slice_idx = int(match_mscmr.group(3))
        return patient_id, phase, slice_idx, None

    return None, None, None, None


def _is_lora_checkpoint(state_dict: Dict[str, Any]) -> bool:
    return any("lora_A" in k for k in state_dict.keys())


def load_checkpoint_and_model(
    checkpoint_path: str,
    bpe_path: str,
    device: str,
    resize_size: int,
    confidence_threshold: float = 0.0,
    use_lora: Optional[bool] = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: float = DEFAULT_LORA_ALPHA,
    lora_target_components: Optional[List[str]] = None,
    lora_unfreeze_components: Optional[List[str]] = None,
    merge_lora_for_inference: bool = True,
):
    """构建 SAM3 + Processor，并加载训练保存的 checkpoint（支持 LoRA 结构）。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    keys = list(state_dict.keys())
    has_detector_prefix = any("detector" in k for k in keys)
    if not has_detector_prefix:
        from collections import OrderedDict

        state_dict = OrderedDict((f"detector.{k}", v) for k, v in state_dict.items())

    if use_lora is None:
        use_lora = _is_lora_checkpoint(state_dict)
        if use_lora:
            logger.info("Checkpoint 含 LoRA 参数，按 LoRA 结构加载。")

    if lora_target_components is None:
        lora_target_components = DEFAULT_LORA_TARGET_COMPONENTS
    if lora_unfreeze_components is None:
        lora_unfreeze_components = (
            DEFAULT_LORA_UNFREEZE_COMPONENTS if use_lora else None
        )

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_components=lora_target_components if use_lora else None,
        lora_freeze_non_lora=False,
        lora_unfreeze_components=lora_unfreeze_components,
    )

    sam3_image_ckpt = {
        k.replace("detector.", ""): v
        for k, v in state_dict.items()
        if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in state_dict.items()
                if "tracker" in k
            }
        )
    missing, unexpected = model.load_state_dict(sam3_image_ckpt, strict=False)
    if missing:
        logger.warning("load_state_dict 缺失键（前 20 个）: %s", list(missing)[:20])
    else:
        logger.info("Checkpoint 权重已加载。")
    if unexpected:
        logger.info("Checkpoint 中未使用的键（已忽略）: %d 个", len(unexpected))

    if use_lora and merge_lora_for_inference:
        merged = merge_lora_into_sam3(model)
        logger.info("推理前已合并 LoRA 权重，共 %d 个线性层。", len(merged))

    processor = Sam3Processor(
        model, resolution=resize_size, confidence_threshold=confidence_threshold
    )
    return model, processor


def build_volume_groups(coco_data: dict, test_dir: str) -> dict:
    """从 COCO JSON 得到按 (patient_id, frame_id) 分组的切片列表。"""
    images_dict = {img["id"]: img for img in coco_data["images"]}

    categories_dict = {}
    categories_names_dict = {}
    for cat in coco_data["categories"]:
        cid = cat["id"]
        if "names" in cat and isinstance(cat["names"], list):
            categories_names_dict[cid] = cat["names"]
            categories_dict[cid] = cat["names"][0]
        elif "name" in cat:
            categories_dict[cid] = cat["name"]
        else:
            raise ValueError(f"Category {cid} must have 'name' or 'names'")

    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    images_by_volume = defaultdict(list)
    for img_id, img_info in images_dict.items():
        patient_id, frame_id, slice_idx, view = extract_patient_frame(
            img_info["file_name"]
        )
        if patient_id is not None and frame_id is not None:
            images_by_volume[(patient_id, frame_id)].append((img_id, slice_idx, view))

    has_view = False
    is_mms2 = False
    for img_list in images_by_volume.values():
        if img_list and len(img_list[0]) >= 3 and img_list[0][2] is not None:
            has_view = True
            if img_list[0][2] in ("SA", "LA"):
                is_mms2 = True
            break

    if has_view and is_mms2:
        filtered = defaultdict(list)
        for key, img_list in images_by_volume.items():
            if img_list and len(img_list[0]) >= 3 and img_list[0][2] == "SA":
                filtered[key] = img_list
        images_by_volume = filtered

    for key in images_by_volume:
        images_by_volume[key].sort(key=lambda x: x[1])

    return {
        "images_dict": images_dict,
        "categories_dict": categories_dict,
        "categories_names_dict": categories_names_dict,
        "annotations_by_image": annotations_by_image,
        "images_by_volume": images_by_volume,
    }


def _safe_category_dir(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").strip() or "unknown"


@torch.inference_mode()
def run_inference_for_volume(
    slice_list,
    volume_categories,
    test_dir,
    images_dict,
    categories_dict,
    categories_names_dict,
    processor,
    save_png_dir=None,
    patient_name=None,
):
    """对单个 volume：按类别、按切片推理，返回每类的 mask 列表。"""
    result_categories = []

    for category_id in volume_categories:
        category_name = categories_dict[category_id]
        if category_id in categories_names_dict:
            text_prompt = np.random.choice(categories_names_dict[category_id]).lower()
        else:
            text_prompt = category_name.lower()

        pred_masks_2d = []
        slice_indices = []

        for img_id, slice_idx, view in slice_list:
            img_info = images_dict[img_id]
            img_path = os.path.join(test_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            img_height = img_info["height"]
            img_width = img_info["width"]

            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )

            masks = inference_state["masks"]
            scores = inference_state["scores"]

            if len(masks) == 0:
                pred_mask_2d = np.zeros((img_height, img_width), dtype=bool)
            else:
                best_idx = scores.argmax().item()
                pred_mask_tensor = masks[best_idx]
                if pred_mask_tensor.dim() == 3:
                    pred_mask_tensor = pred_mask_tensor.squeeze(0)
                pred_mask_2d = pred_mask_tensor.cpu().numpy().astype(bool)
                if (
                    pred_mask_2d.shape[0] != img_height
                    or pred_mask_2d.shape[1] != img_width
                ):
                    pred_pil = Image.fromarray(pred_mask_2d.astype(np.uint8) * 255)
                    pred_pil = pred_pil.resize((img_width, img_height), Image.NEAREST)
                    pred_mask_2d = np.array(pred_pil) > 0

            pred_masks_2d.append(pred_mask_2d)
            slice_indices.append(slice_idx)

            if save_png_dir and patient_name:
                png_subdir = os.path.join(
                    save_png_dir, patient_name, _safe_category_dir(category_name)
                )
                os.makedirs(png_subdir, exist_ok=True)
                png_path = os.path.join(png_subdir, f"slice_{slice_idx:03d}.png")
                Image.fromarray(pred_mask_2d.astype(np.uint8) * 255).save(png_path)

        if not pred_masks_2d:
            continue

        result_categories.append({
            "category_id": category_id,
            "category_name": category_name,
            "slice_indices": slice_indices,
            "masks": pred_masks_2d,
        })

    return result_categories


def save_volume_as_nii(categories_result, patient_name, nii_dir, affine=None):
    """将单 volume 多类别预测合并为 3D 标签体并保存 .nii.gz（需 nibabel）。"""
    if not HAS_NIBABEL:
        raise RuntimeError("保存 nii.gz 需要: pip install nibabel")
    if not categories_result:
        return
    sorted_cats = sorted(categories_result, key=lambda x: x["category_id"])
    n_slices = len(sorted_cats[0]["masks"])
    h, w = sorted_cats[0]["masks"][0].shape
    vol = np.zeros((n_slices, h, w), dtype=np.uint8)
    for label_val, cat in enumerate(sorted_cats, start=1):
        for i, mask_2d in enumerate(cat["masks"]):
            if i < vol.shape[0] and mask_2d.shape[0] == h and mask_2d.shape[1] == w:
                vol[i][mask_2d] = label_val
    vol = np.transpose(vol, (2, 1, 0))
    if affine is None:
        affine = np.eye(4)
    nii = nib.Nifti1Image(vol, affine, nib.Nifti1Header())
    safe_name = _safe_category_dir(patient_name)
    out_path = os.path.join(nii_dir, f"{safe_name}.nii.gz")
    os.makedirs(nii_dir, exist_ok=True)
    nib.save(nii, out_path)


def _default_test_dir() -> str:
    return (
        "/home/gaoqi/dataset/using/acdc/processed/png_coco_sam3_fullframes_weak/test"
    )


def _default_checkpoint() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/scribble_tmi_lora/checkpoints/"
        "val_acdc_segmentation_coco_eval_segm_AP.pt"
    )


def _default_output_dir(test_dir: str) -> str:
    parent = os.path.dirname(test_dir.rstrip(os.sep))
    return os.path.join(parent, "inference_predictions")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 批量推理：按 volume/类别/切片保存 predictions.pkl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="详见模块顶部文档字符串。",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=_default_test_dir(),
        help="测试集目录（内含 image_annotations.coco.json 与 images/）",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=_default_checkpoint(),
        help="训练保存的 .pt（含 model / optimizer 等，本脚本只读 model）",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="image_annotations.coco.json",
        help="COCO JSON 文件名（位于 test_dir 下）",
    )
    parser.add_argument("--resize_size", type=int, default=1008, help="Processor 输入边长")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Sam3Processor 置信度过滤阈值；指标异常时可试 0.0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录；默认 test 的上一级目录下的 inference_predictions",
    )
    parser.add_argument(
        "--output_name", type=str, default="predictions.pkl", help="输出 pkl 文件名"
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="另存每切片 mask 为 PNG（output_dir/pngs/...）",
    )
    parser.add_argument(
        "--png_dir",
        type=str,
        default=None,
        help="PNG 目录；默认 output_dir/pngs",
    )
    parser.add_argument(
        "--save_nii",
        action="store_true",
        help="另存每 volume 为 .nii.gz（需 nibabel）",
    )
    parser.add_argument(
        "--nii_dir",
        type=str,
        default=None,
        help="NIfTI 目录；默认 output_dir/nii",
    )
    parser.add_argument(
        "--use_lora",
        type=lambda x: None if x.lower() == "auto" else x.lower() == "true",
        default=None,
        metavar="auto|true|false",
        help="是否按 LoRA 构建模型；默认 auto（根据权重是否含 lora_A）",
    )
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument(
        "--merge_lora_for_inference",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="true|false",
        help="是否在推理前把 LoRA 权重合并进基础线性层；默认 true。",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    json_path = os.path.join(args.test_dir, args.annotation_file)

    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.test_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    save_png_dir = None
    if args.save_png:
        save_png_dir = args.png_dir or os.path.join(args.output_dir, "pngs")
        os.makedirs(save_png_dir, exist_ok=True)
        logger.info("PNG 输出: %s", save_png_dir)

    save_nii_dir = None
    if args.save_nii:
        if not HAS_NIBABEL:
            raise RuntimeError("--save_nii 需要: pip install nibabel")
        save_nii_dir = args.nii_dir or os.path.join(args.output_dir, "nii")
        os.makedirs(save_nii_dir, exist_ok=True)
        logger.info("NIfTI 输出: %s", save_nii_dir)

    logger.info("加载模型 …")
    model, processor = load_checkpoint_and_model(
        args.checkpoint_path,
        bpe_path,
        device,
        args.resize_size,
        args.confidence_threshold,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        merge_lora_for_inference=args.merge_lora_for_inference,
    )

    logger.info("读取 COCO: %s", json_path)
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    data = build_volume_groups(coco_data, args.test_dir)
    images_dict = data["images_dict"]
    categories_dict = data["categories_dict"]
    categories_names_dict = data["categories_names_dict"]
    annotations_by_image = data["annotations_by_image"]
    images_by_volume = data["images_by_volume"]

    volume_keys = sorted(images_by_volume.keys())
    volumes_out = []

    for patient_id, frame_id in tqdm(volume_keys, desc="Volumes"):
        slice_list = images_by_volume[(patient_id, frame_id)]
        if not slice_list:
            continue

        first_view = slice_list[0][2]
        if frame_id == "default":
            first_img_info = images_dict[slice_list[0][0]]
            first_filename = first_img_info["file_name"]
            if first_filename.startswith("Case"):
                patient_name = f"Case{patient_id}"
            else:
                patient_name = patient_id
        elif first_view is None:
            # MSCMR: subject10_DE_slice000.png -> patient_name = subject10_DE
            if isinstance(frame_id, str) and not frame_id.isdigit():
                patient_name = f"subject{patient_id}_{frame_id}"
            else:
                patient_name = f"patient{patient_id}_frame{frame_id}"
        elif first_view in ("SA", "LA"):
            patient_name = f"{patient_id}_{frame_id}"
        else:
            patient_name = f"patient{patient_id}_{frame_id}"

        volume_categories = set()
        for img_id, _, _ in slice_list:
            for ann in annotations_by_image[img_id]:
                volume_categories.add(ann["category_id"])
        if not volume_categories:
            continue

        categories_result = run_inference_for_volume(
            slice_list,
            volume_categories,
            args.test_dir,
            images_dict,
            categories_dict,
            categories_names_dict,
            processor,
            save_png_dir=save_png_dir,
            patient_name=patient_name if save_png_dir else None,
        )

        if save_nii_dir and categories_result:
            try:
                save_volume_as_nii(
                    categories_result,
                    patient_name,
                    save_nii_dir,
                )
            except Exception as e:
                tqdm.write(f"Warning: nii save failed for {patient_name}: {e}")

        volumes_out.append({
            "patient_id": patient_id,
            "frame_id": frame_id,
            "patient_name": patient_name,
            "categories": categories_result,
        })

    result = {
        "volumes": volumes_out,
        "config": {
            "test_dir": args.test_dir,
            "annotation_file": args.annotation_file,
            "checkpoint_path": args.checkpoint_path,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    logger.info("已写入 %s（%d volumes）", out_path, len(volumes_out))


if __name__ == "__main__":
    main()
