"""
SAM3 模型批量推理脚本 - 3D 医学图像分割

对测试集按 patient+frame 分组，对每个 volume 的每个类别、每个切片运行 SAM3 推理，
将预测 mask 保存到文件，供 batch_evaluate.py 离线计算指标。

预测逻辑：
- Sam3Processor 会先按 confidence_threshold 过滤，只保留置信度不低于阈值的 mask。
- 在满足阈值的候选中，取置信度最高的一个 mask 作为该切片的预测结果（scores.argmax()）。
- 若过滤后无候选，则输出全零 mask。

功能：
- 从 COCO 格式 JSON 读取图像列表与类别
- 按 patient+frame 分组（支持 ACDC / MMs2 / CAMUS / BTCV / Promise12 命名）
- 加载 SAM3 模型与 processor，对每张切片做 text-prompt 分割
- 将每 volume 每类别的 2D 预测 mask 列表及 slice_indices 保存为 .pkl
- 可选：将每个 volume 的预测合并为 3D 标签体并保存为 .nii.gz（一个样本一个文件）

LoRA 微调 checkpoint 支持（修改说明）：
- LoRA 微调保存的 checkpoint 中，原 nn.Linear 被替换为 LoRALinear，state_dict 的 key
  形如 backbone....qkv.linear.weight、backbone....qkv.lora_A / lora_B，与未注入 LoRA 的
  模型（期望 qkv.weight）不一致。若用非 LoRA 模型加载 LoRA checkpoint，会大量缺失 key，
  导致效果极差。
- 本脚本在加载 checkpoint 时会自动检测是否为 LoRA 微调保存的（通过 state_dict 中是否
  含有 "lora_A" 的 key）。若为 LoRA checkpoint，则使用与 train_lora.yaml 一致的 LoRA
  配置构建模型（use_lora=True, lora_r=8, lora_alpha=16.0，以及相同的 lora_target_components），
  再加载权重，保证 key 一一对应。
- 也可通过命令行显式指定：--use_lora true 强制按 LoRA 加载，--use_lora false 强制按
  普通模型加载；--lora_r、--lora_alpha 需与训练时一致（默认 8、16.0）。
- 推理阶段构建 LoRA 模型时使用 lora_freeze_non_lora=False，不冻结参数。

用法：
    python gq_scripts/evaluate/batch_inference.py \
        --test_dir /home/gaoqi/dataset/using/acdc4/test \
        --checkpoint_path /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
        --confidence_threshold 0.7 \
        --output_dir /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_acdc_0.7 \
        --save_nii

    CUDA_VISIBLE_DEVICES=2 python gq_scripts/evaluate/batch_inference.py \
        --test_dir /home/gaoqi/dataset/using/btcv_2/test \
        --checkpoint_path /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
        --confidence_threshold 0.7 \
        --output_dir /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_btcv_0.7

    CUDA_VISIBLE_DEVICES=4 python gq_scripts/evaluate/batch_inference.py \
        --test_dir /home/gaoqi/dataset/using/promise12_3/test \
        --checkpoint_path /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
        --confidence_threshold 0.7 \
        --output_dir /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_promise12_0.7

    # 同时保存 .nii.gz（每个 volume 一个文件，便于与 GT nii 对比）
    python gq_scripts/evaluate/batch_inference.py --test_dir ... --checkpoint_path ... --output_dir ... --save_nii

示例：
    # 普通 / LoRA checkpoint 均可，脚本会自动识别 LoRA
    python batch_inference.py --test_dir /home/gaoqi/dataset/using/acdc4/test \\
        --checkpoint_path /path/to/checkpoint.pt --output_dir ./inference_output

    # 显式指定按 LoRA 加载（训练时若改过 lora_r/lora_alpha 请一并指定）
    python batch_inference.py --test_dir /path/to/test --checkpoint_path /path/to/lora.pt \\
        --use_lora true --lora_r 8 --lora_alpha 16.0
"""

import os
import re
import json
import argparse
import pickle
from collections import defaultdict

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
from sam3.model.sam3_image_processor import Sam3Processor


def extract_patient_frame(img_file):
    """从文件名提取 patient_id、frame_id 和 slice_idx。

    支持：
    - ACDC: patient{数字}_frame{数字}_slice{数字}.png
    - MMs2: {patient_id}_{view}_{phase}_slice{数字}.png (view: SA/LA, phase: ED/ES)
    - CAMUS: patient{数字}_{view}_{phase}_slice{数字}.png 或 无 _slice 后缀
    - BTCV: {patient_id}-Image_slice{数字}.png (例如: 0507688-Image_slice000.png)
    - Promise12: Case{patient_id}_slice{数字}.png (例如: Case00_slice000.png)

    Returns:
        (patient_id, frame_id, slice_idx, view) 或 (None, None, None, None)
    """
    basename = os.path.basename(img_file)

    match_acdc = re.match(r"patient(\d+)_frame(\d+)_slice(\d+)\.png", basename)
    if match_acdc:
        patient_id = match_acdc.group(1)
        frame_id = match_acdc.group(2)
        slice_idx = int(match_acdc.group(3))
        return patient_id, frame_id, slice_idx, None

    match_mms2 = re.match(r"(\d+)_(SA|LA)_(ED|ES)_slice(\d+)\.png", basename)
    if match_mms2:
        patient_id = match_mms2.group(1)
        view = match_mms2.group(2)
        phase = match_mms2.group(3)
        slice_idx = int(match_mms2.group(4))
        frame_id = f"{view}_{phase}"
        return patient_id, frame_id, slice_idx, view

    match_camus = re.match(
        r"patient(\d+)_([A-Z0-9]+)_(ED|ES)(?:_slice(\d+))?\.png", basename
    )
    if match_camus:
        patient_id = match_camus.group(1)
        view = match_camus.group(2)
        phase = match_camus.group(3)
        slice_idx_str = match_camus.group(4)
        slice_idx = int(slice_idx_str) if slice_idx_str else 0
        frame_id = f"{view}_{phase}"
        return patient_id, frame_id, slice_idx, view

    # BTCV: {patient_id}-Image_slice{slice_idx}.png (例如: 0507688-Image_slice000.png)
    match_btcv = re.match(r"([\d]+)-Image_slice(\d+)\.png", basename)
    if match_btcv:
        patient_id = match_btcv.group(1)
        slice_idx = int(match_btcv.group(2))
        frame_id = "default"  # BTCV 没有 frame 概念，使用默认值
        return patient_id, frame_id, slice_idx, None

    # Promise12: Case{patient_id}_slice{slice_idx}.png (例如: Case00_slice000.png)
    match_promise12 = re.match(r"Case(\d+)_slice(\d+)\.png", basename)
    if match_promise12:
        patient_id = match_promise12.group(1)
        slice_idx = int(match_promise12.group(2))
        frame_id = "default"  # Promise12 没有 frame 概念，使用默认值
        return patient_id, frame_id, slice_idx, None

    return None, None, None, None


# 与 train_lora.yaml 一致的 LoRA 配置，用于加载 LoRA 微调后的 checkpoint
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16.0
DEFAULT_LORA_TARGET_COMPONENTS = [
    "vision_encoder",
    "text_encoder",
    "geometry_encoder",
    "detr_encoder",
    "detr_decoder",
    "mask_decoder",
]


def _is_lora_checkpoint(state_dict):
    """根据 state_dict 的 key 判断是否为 LoRA 微调保存的 checkpoint。"""
    return any("lora_A" in k for k in state_dict.keys())


def load_checkpoint_and_model(
    checkpoint_path,
    bpe_path,
    device,
    resize_size,
    confidence_threshold=0.0,
    use_lora=None,
    lora_r=DEFAULT_LORA_R,
    lora_alpha=DEFAULT_LORA_ALPHA,
    lora_target_components=None,
):
    """加载检查点并构建 SAM3 模型与 Processor。

    若 checkpoint 为 LoRA 微调保存的，必须用 use_lora=True 构建模型（或由内部自动检测）。
    use_lora: True/False 显式指定；None 时根据 checkpoint 中是否含 lora_A 自动检测。
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    keys = list(state_dict.keys())
    has_detector_prefix = any("detector" in k for k in keys)

    if not has_detector_prefix:
        from collections import OrderedDict
        processed = OrderedDict()
        for k, v in state_dict.items():
            processed["detector." + k] = v
        state_dict = processed

    # 判断是否为 LoRA checkpoint，决定是否用 LoRA 结构构建模型
    if use_lora is None:
        use_lora = _is_lora_checkpoint(state_dict)
        if use_lora:
            print("检测到 LoRA 微调 checkpoint，将使用 LoRA 模型结构加载。")
    if lora_target_components is None:
        lora_target_components = DEFAULT_LORA_TARGET_COMPONENTS

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
    )

    sam3_image_ckpt = {
        k.replace("detector.", ""): v
        for k, v in state_dict.items()
        if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update({
            k.replace("tracker.", "inst_interactive_predictor.model."): v
            for k, v in state_dict.items()
            if "tracker" in k
        })
    missing, unexpected = model.load_state_dict(sam3_image_ckpt, strict=False)
    if missing:
        print(f"加载检查点时缺失的键: {missing}")
    else:
        print("检查点加载成功")
    if unexpected:
        print(f"检查点中多余的键（已忽略）: {unexpected}")

    processor = Sam3Processor(model, resolution=resize_size, confidence_threshold=confidence_threshold)
    return model, processor


def build_volume_groups(coco_data, test_dir):
    """从 COCO JSON 构建按 (patient_id, frame_id) 分组的切片列表。"""
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

    # MMs2: 只保留 SA 视图
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


def _safe_category_dir(name):
    """将类别名转为可作目录名的字符串（替换空格等）。"""
    return name.replace(" ", "_").replace("/", "_").strip() or "unknown"


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
    """对单个 volume 的所有类别、所有切片运行推理，返回按类别组织的预测 mask。"""
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

            image = Image.open(img_path)
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
                if pred_mask_2d.shape[0] != img_height or pred_mask_2d.shape[1] != img_width:
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
    """将单个 volume 的按类别预测合并为 3D 标签体并保存为 .nii.gz。

    标签规则：0=背景，1=第一个类别(category_id 最小)，2=第二个类别，以此类推。
    内存中先按 (n_slices, H, W) 构建，保存时转为 (W, H, n_slices) 以与常见 NIfTI/ITK-SNAP
    主图维度 (x, y, z) = (W, H, n_slices) 一致，便于叠加显示。

    Args:
        categories_result: run_inference_for_volume 返回的列表，每项含 category_id, masks, slice_indices
        patient_name: 用于文件名的病例标识，如 patient101_frame01 或 Case00
        nii_dir: 输出目录
        affine: 可选，3D 仿射矩阵；默认使用单位矩阵（无 spacing 信息时）
    """
    if not HAS_NIBABEL:
        raise RuntimeError("保存 nii.gz 需要安装 nibabel: pip install nibabel")
    if not categories_result:
        return
    # 按 category_id 排序，保证标签顺序一致（1, 2, 3, ...）
    sorted_cats = sorted(categories_result, key=lambda x: x["category_id"])
    n_slices = len(sorted_cats[0]["masks"])
    h, w = sorted_cats[0]["masks"][0].shape
    vol = np.zeros((n_slices, h, w), dtype=np.uint8)
    for label_val, cat in enumerate(sorted_cats, start=1):
        for i, mask_2d in enumerate(cat["masks"]):
            if i < vol.shape[0] and mask_2d.shape[0] == h and mask_2d.shape[1] == w:
                vol[i][mask_2d] = label_val
    # 转为 (W, H, n_slices) 使 NIfTI 维度与主图 (x, y, z) 一致，ITK-SNAP 可正确叠加
    vol = np.transpose(vol, (2, 1, 0))  # (n_slices, H, W) -> (W, H, n_slices)
    if affine is None:
        affine = np.eye(4)
    nii = nib.Nifti1Image(vol, affine, nib.Nifti1Header())
    safe_name = _safe_category_dir(patient_name)
    out_path = os.path.join(nii_dir, f"{safe_name}.nii.gz")
    os.makedirs(nii_dir, exist_ok=True)
    nib.save(nii, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 批量推理：按 volume/类别/切片运行推理并保存预测 mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/home/gaoqi/dataset/using/acdc5/test",
        help="测试集目录",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/acdc/lora/checkpoints/val_acdc_segmentation_coco_eval_segm_AP_merged.pt",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="image_annotations.coco.json",
        help="COCO 标注 JSON 文件名",
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=1008,
        help="模型输入尺寸",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Sam3Processor 置信度阈值，低于此分数的预测将被过滤。评估时若 Dice/IoU 全为 0 可尝试设为 0.0（默认 0.0）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="推理结果输出目录；默认与 test_dir 同级的 inference_predictions",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="predictions.pkl",
        help="输出文件名（默认 predictions.pkl）",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="同时将每张切片的预测 mask 保存为 PNG，目录为 output_dir/pngs/<patient_name>/<category>/slice_XXX.png",
    )
    parser.add_argument(
        "--png_dir",
        type=str,
        default=None,
        help="PNG 保存目录；未指定且 --save_png 时使用 output_dir/pngs",
    )
    parser.add_argument(
        "--save_nii",
        action="store_true",
        help="同时将每个 volume 的预测保存为 .nii.gz（一个样本一个文件），需安装 nibabel",
    )
    parser.add_argument(
        "--nii_dir",
        type=str,
        default=None,
        help="NIfTI 保存目录；未指定且 --save_nii 时使用 output_dir/nii",
    )
    parser.add_argument(
        "--use_lora",
        type=lambda x: x.lower() == "true",
        default=None,
        metavar="true|false",
        help="是否按 LoRA 模型加载（与 train_lora.yaml 一致）。默认根据 checkpoint 是否含 lora_A 自动检测",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=DEFAULT_LORA_R,
        help=f"LoRA rank，需与训练时一致（默认 {DEFAULT_LORA_R}）",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=DEFAULT_LORA_ALPHA,
        help=f"LoRA alpha，需与训练时一致（默认 {DEFAULT_LORA_ALPHA}）",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    json_path = os.path.join(args.test_dir, args.annotation_file)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.test_dir.rstrip("/")), "inference_predictions"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    save_png_dir = None
    if args.save_png:
        save_png_dir = args.png_dir or os.path.join(args.output_dir, "pngs")
        os.makedirs(save_png_dir, exist_ok=True)
        print(f"Will save per-slice PNGs to: {save_png_dir}")

    save_nii_dir = None
    if args.save_nii:
        if not HAS_NIBABEL:
            raise RuntimeError("--save_nii 需要安装 nibabel，请运行: pip install nibabel")
        save_nii_dir = args.nii_dir or os.path.join(args.output_dir, "nii")
        os.makedirs(save_nii_dir, exist_ok=True)
        print(f"Will save per-volume NIfTI to: {save_nii_dir}")

    print("Loading model ...")
    model, processor = load_checkpoint_and_model(
        args.checkpoint_path,
        bpe_path,
        device,
        args.resize_size,
        args.confidence_threshold,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    print("Loading COCO JSON ...")
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
        # BTCV 和 Promise12 使用 frame_id="default"，需要特殊处理
        if frame_id == "default":
            # 检查是否为 Promise12 (CaseXX 格式) 或 BTCV (数字-Image 格式)
            first_img_info = images_dict[slice_list[0][0]]
            first_filename = first_img_info["file_name"]
            if first_filename.startswith("Case"):
                # Promise12: Case00_slice000.png -> Case00
                patient_name = f"Case{patient_id}"
            else:
                # BTCV: 0507688-Image_slice000.png -> 0507688
                patient_name = patient_id
        elif first_view is None:
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
                tqdm.write(f"Warning: failed to save nii for {patient_name}: {e}")

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
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Predictions saved to {out_path} ({len(volumes_out)} volumes)")


if __name__ == "__main__":
    main()
