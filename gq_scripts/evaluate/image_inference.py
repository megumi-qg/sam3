import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.eval.postprocessors import PostProcessImage

import torch
import torchvision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#########################
##   MAIN  sam3 图像推理脚本
#########################

def process_single_image(image, processor, prompt, image_path, output_dir, slice_idx=None):
    """
    对单张图像进行推理
    
    Args:
        image: PIL Image对象或numpy数组
        processor: Sam3Processor实例
        prompt: 文本提示
        image_path: 图像路径（用于生成输出文件名）
        output_dir: 输出目录
        slice_idx: 切片索引（如果是npz文件，用于生成文件名）
    
    Returns:
        inference_state: 推理状态字典
    """
    # 确保图像是PIL Image
    if isinstance(image, np.ndarray):
        # 如果是灰度图像，转换为RGB
        if len(image.shape) == 2:
            image = Image.fromarray(image, mode='L').convert('RGB')
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = Image.fromarray(image.squeeze(2), mode='L').convert('RGB')
        else:
            image = Image.fromarray(image)
    
    width, height = image.size
    print(f"处理图像: width={width}, height={height}" + (f", slice_idx={slice_idx}" if slice_idx is not None else ""))
    
    # 设置图像
    inference_state = processor.set_image(image)
    
    # 推理
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    masks, boxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]
    print(f"检测到 {len(scores)} 个对象")
    
    # 可视化结果
    plt.close("all")
    plot_results(image, inference_state)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    if slice_idx is not None:
        # npz文件的切片
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"inference_result_{base_name}_slice{slice_idx:03d}.png")
    else:
        # 普通图像文件
        output_path = os.path.join(output_dir, f"inference_result_{os.path.basename(image_path)}.png")
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存结果到: {output_path}")
    
    return inference_state

def main():
    parser = argparse.ArgumentParser(description='SAM3图像推理脚本')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/acdc_camus/full/1/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt",
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--image',
        type=str,
        default="/home/gaoqi/dataset/using/acdc4/test/data/patient110_frame01.npz",
        help='图像路径（PNG/JPG文件或NPZ文件）'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="Right Ventricle",
        help='文本提示（例如："right ventricle"）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认为脚本目录下的outputs文件夹）'
    )
    parser.add_argument(
        '--resize_size',
        type=int,
        default=1008,
        help='图像resize大小（默认：1008）'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.75,
        help='置信度阈值（默认：0.1）'
    )
    
    args = parser.parse_args()
    
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    print("sam3_root:", sam3_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # - Build model
    print("Loading model ...")
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = args.checkpoint

    # 预处理检查点：处理 model 键和 detector 前缀
    print(f"预处理检查点文件: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 提取 model 键（如果存在）
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # 检查键是否包含 "detector" 前缀
    keys = list(state_dict.keys())
    has_detector_prefix = any("detector" in k for k in keys)

    if not has_detector_prefix:
        print("检测到键没有 'detector.' 前缀，自动添加...")
        from collections import OrderedDict
        processed_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = "detector." + k
            processed_state_dict[new_key] = v
        state_dict = processed_state_dict
        print("已添加 'detector.' 前缀")
    else:
        print("检查点已包含 'detector.' 前缀，无需处理")

    # 构建模型（不加载检查点）
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=None,  # 不通过文件路径加载
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    # 手动加载处理后的检查点
    print("加载检查点到模型...")
    # 按照 _load_checkpoint 的逻辑处理检查点
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in state_dict.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in state_dict.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(f"加载检查点时发现缺失的键: {missing_keys}")
    else:
        print("检查点加载成功！")

    # 创建processor
    print("创建图像处理器...")
    processor = Sam3Processor(model, resolution=args.resize_size, confidence_threshold=args.confidence_threshold)
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # - Load image or npz
    image_path = args.image
    prompt = args.prompt
    
    # 检查是否为npz文件
    if image_path.endswith(".npz"):
        print(f"检测到NPZ文件: {image_path}")
        print("正在加载NPZ数据...")
        
        # 加载npz文件
        npz_data = np.load(image_path)
        
        # 检查必需的键
        required_keys = ['imgs', 'gts', 'spacing']
        missing_keys = [k for k in required_keys if k not in npz_data]
        if missing_keys:
            raise ValueError(f"NPZ文件缺少必需的键: {missing_keys}")
        
        imgs = npz_data['imgs']  # (D, H, W), uint8
        gts = npz_data['gts']    # (D, H, W), int32
        spacing = npz_data['spacing']  # (depth, height, width)
        
        print(f"NPZ数据形状: imgs={imgs.shape}, gts={gts.shape}, spacing={spacing}")
        print(f"将对 {imgs.shape[0]} 个切片进行推理...")
        
        # 对每个切片进行推理
        for slice_idx in range(imgs.shape[0]):
            print(f"\n{'='*50}")
            print(f"处理切片 {slice_idx + 1}/{imgs.shape[0]}")
            print(f"{'='*50}")
            
            # 获取单个切片 (H, W)
            slice_img = imgs[slice_idx]
            
            # 确保是uint8类型
            if slice_img.dtype != np.uint8:
                slice_img = slice_img.astype(np.uint8)
            
            # 转换为PIL Image（灰度转RGB）
            img_pil = Image.fromarray(slice_img, mode='L').convert('RGB')
            
            # 进行推理
            process_single_image(
                image=img_pil,
                processor=processor,
                prompt=prompt,
                image_path=image_path,
                output_dir=output_dir,
                slice_idx=slice_idx
            )
        
        print(f"\n{'='*50}")
        print(f"已完成所有 {imgs.shape[0]} 个切片的推理！")
        print(f"{'='*50}")
    else:
        # 普通图像文件
        print(f"加载图像: {image_path}")
        image = Image.open(image_path)
        
        # 进行推理
        process_single_image(
            image=image,
            processor=processor,
            prompt=prompt,
            image_path=image_path,
            output_dir=output_dir,
            slice_idx=None
        )

if __name__ == "__main__":
    main()