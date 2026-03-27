"""
将 LoRA 微调后的 checkpoint 合并为普通权重：把 delta_W = (alpha/r) * B @ A 加回原始 W，
并把 LoRALinear 替换为 nn.Linear，保存后的模型可用标准 build_sam3_image_model(use_lora=False) 加载推理。

公式：W_merged = W + (lora_alpha / r) * (B.T @ A.T)，其中 W、A、B 来自 LoRALinear。

用法：
    python merge_lora_ckpt.py --input_ckpt <LoRA 检查点路径> --output_ckpt <合并后保存路径> [--lora_r 8] [--lora_alpha 16.0]

示例：
    python gq_scripts/tool/merge_lora_ckpt.py \
        --input_ckpt /home/gaoqi/sam3/gq_experiment/final/lora_acdc_btcv_promise12/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
        --output_ckpt /home/gaoqi/sam3/gq_experiment/final/lora_acdc_btcv_promise12/checkpoints/sam3_full_fintuned.pt
"""

import os
import sys
import argparse

import torch

# 保证可导入 sam3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sam3 import build_sam3_image_model
from sam3.model.lora import merge_lora_into_sam3

# 与 train_lora.yaml 一致
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


def main():
    parser = argparse.ArgumentParser(
        description="将 LoRA checkpoint 合并为普通权重（W_merged = W + (alpha/r)*B@A）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_ckpt",
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/acdc/lora/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt",
        help="LoRA 微调保存的检查点路径",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default=None,
        help="合并后保存路径；默认在 input 同目录下，文件名加 _merged",
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
    parser.add_argument(
        "--bpe_path",
        type=str,
        default=None,
        help="BPE 词表路径；默认使用 sam3 包内 assets",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_ckpt):
        print(f"错误: 找不到输入检查点 {args.input_ckpt}")
        sys.exit(1)

    if args.output_ckpt is None:
        base, ext = os.path.splitext(args.input_ckpt)
        args.output_ckpt = base + "_merged" + ext

    if args.bpe_path is None:
        import sam3
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        args.bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    device = "cpu"
    print("加载 LoRA 检查点 ...")
    ckpt = torch.load(args.input_ckpt, map_location=device, weights_only=False)

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    has_detector = any("detector" in k for k in state_dict.keys())
    if not has_detector:
        from collections import OrderedDict
        state_dict = OrderedDict(("detector." + k, v) for k, v in state_dict.items())

    print("构建 LoRA 模型并加载权重 ...")
    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_components=DEFAULT_LORA_TARGET_COMPONENTS,
        lora_freeze_non_lora=False,
    )

    sam3_image_ckpt = {
        k.replace("detector.", ""): v
        for k, v in state_dict.items()
        if "detector" in k
    }
    missing, unexpected = model.load_state_dict(sam3_image_ckpt, strict=False)
    if missing:
        print(f"警告: 加载时缺失的键数量: {len(missing)}")
    if unexpected:
        print(f"警告: 检查点中多余的键数量: {len(unexpected)}")

    print("合并 LoRA 到原始 W (W_merged = W + (alpha/r)*B.T@A.T) ...")
    merged_paths = merge_lora_into_sam3(model)
    print(f"已合并 {len(merged_paths)} 个 LoRALinear 层")

    merged_state = model.state_dict()
    out_state = {"model": {"detector." + k: v for k, v in merged_state.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)) or ".", exist_ok=True)
    torch.save(out_state, args.output_ckpt)
    print(f"已保存合并后的检查点到: {args.output_ckpt}")
    print("推理时可直接用 build_sam3_image_model(use_lora=False) 加载该文件。")


if __name__ == "__main__":
    main()
