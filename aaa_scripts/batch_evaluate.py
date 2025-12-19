import os
import json
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from pycocotools import mask as mask_utils
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def compute_iou(pred_mask, gt_mask):
    """计算IoU (Intersection over Union)"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """计算Dice系数"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    if pred_area + gt_area == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / (pred_area + gt_area)


def decode_rle_mask(rle, height, width):
    """解码COCO RLE格式的mask"""
    if isinstance(rle, dict):
        rle_obj = rle
    else:
        rle_obj = {
            "counts": rle,
            "size": [height, width]
        }
    mask = mask_utils.decode(rle_obj)
    return mask.astype(bool)


def main():
    # 配置路径
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据集路径
    test_dir = "/home/gaoqi/sam3/dataset/ACDC_new/test"
    json_path = os.path.join(test_dir, "image_annotations.coco.json")
    
    # 模型配置
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = "/home/gaoqi/sam3/experiment/acdc_weak_finetune_3/checkpoints/checkpoint.pt"
    resize_size = 1008
    
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
    
    # 加载模型（不通过文件路径加载检查点）
    print("Loading model ...")
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
    
    processor = Sam3Processor(model, resolution=resize_size, confidence_threshold=0.0)
    
    # 读取COCO JSON文件
    print("Loading COCO JSON file ...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建索引
    images_dict = {img['id']: img for img in coco_data['images']}
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 按图像ID组织标注
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # 统计结果
    category_ious = defaultdict(list)
    category_dices = defaultdict(list)
    all_ious = []
    all_dices = []
    
    # 遍历所有图像
    print("Starting batch evaluation ...")
    image_ids = sorted(images_dict.keys())
    
    for img_id in tqdm(image_ids, desc="Processing images"):
        img_info = images_dict[img_id]
        img_file = img_info['file_name']
        img_path = os.path.join(test_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # 加载图像
        image = Image.open(img_path)
        img_height, img_width = img_info['height'], img_info['width']
        
        # 设置图像
        inference_state = processor.set_image(image)
        
        # 获取该图像的所有标注（只包含该图像实际存在的标注，不是所有类别）
        annotations = annotations_by_image[img_id]
        
        if len(annotations) == 0:
            # 如果该图像没有任何标注，跳过
            continue
        
        # 对每个标注条目进行推理（根据JSON文件中的具体标注，不是对所有类别都推理）
        for ann in annotations:
            category_id = ann['category_id']
            category_name = categories_dict[category_id]
            
            # 解码ground truth mask（从JSON文件中的segmentation字段）
            segm = ann['segmentation']
            gt_mask = decode_rle_mask(segm, img_height, img_width)
            
            # 使用该标注对应的类别名称作为prompt进行推理
            # 注意：只对JSON文件中实际存在的标注进行推理，不会对所有类别都推理
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, 
                prompt=category_name.lower()
            )
            
            # 获取推理结果
            masks = inference_state["masks"]
            scores = inference_state["scores"]
            
            if len(masks) == 0:
                # 没有检测到任何mask，IoU和Dice都为0
                iou = 0.0
                dice = 0.0
            else:
                # 选择score最高的mask
                best_idx = scores.argmax().item()
                pred_mask_tensor = masks[best_idx]
                
                # mask的形状可能是 (1, H, W) 或 (H, W)
                if pred_mask_tensor.dim() == 3:
                    pred_mask_tensor = pred_mask_tensor.squeeze(0)
                
                # 转换为numpy数组
                pred_mask_np = pred_mask_tensor.cpu().numpy().astype(bool)
                
                # 如果尺寸不匹配，需要resize到原始图像尺寸
                if pred_mask_np.shape != gt_mask.shape:
                    pred_mask_pil = Image.fromarray(pred_mask_np.astype(np.uint8) * 255)
                    pred_mask_pil = pred_mask_pil.resize((img_width, img_height), Image.NEAREST)
                    pred_mask_np = np.array(pred_mask_pil) > 0
                
                # 计算IoU和Dice
                iou = compute_iou(pred_mask_np, gt_mask)
                dice = compute_dice(pred_mask_np, gt_mask)
            
            # 记录结果
            category_ious[category_name].append(iou)
            category_dices[category_name].append(dice)
            all_ious.append(iou)
            all_dices.append(dice)
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nPer-category metrics:")
    print("-"*60)
    print(f"{'Category':<20} {'IoU':<15} {'Dice':<15} {'Count':<10}")
    print("-"*60)
    
    for category_name in sorted(category_ious.keys()):
        ious = category_ious[category_name]
        dices = category_dices[category_name]
        mean_iou = np.mean(ious)
        mean_dice = np.mean(dices)
        count = len(ious)
        print(f"{category_name:<20} {mean_iou:<15.4f} {mean_dice:<15.4f} {count:<10}")
    
    print("-"*60)
    print(f"{'Overall':<20} {np.mean(all_ious):<15.4f} {np.mean(all_dices):<15.4f} {len(all_ious):<10}")
    print("="*60)
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "evaluation_results.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write("Per-category metrics:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Category':<20} {'IoU':<15} {'Dice':<15} {'Count':<10}\n")
        f.write("-"*60 + "\n")
        
        for category_name in sorted(category_ious.keys()):
            ious = category_ious[category_name]
            dices = category_dices[category_name]
            mean_iou = np.mean(ious)
            mean_dice = np.mean(dices)
            count = len(ious)
            f.write(f"{category_name:<20} {mean_iou:<15.4f} {mean_dice:<15.4f} {count:<10}\n")
        
        f.write("-"*60 + "\n")
        f.write(f"{'Overall':<20} {np.mean(all_ious):<15.4f} {np.mean(all_dices):<15.4f} {len(all_ious):<10}\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

