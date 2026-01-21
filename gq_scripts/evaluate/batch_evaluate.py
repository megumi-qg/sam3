"""
SAM3模型批量评估脚本 - 3D医学图像分割评估

本脚本用于评估SAM3模型在医学图像分割任务（如ACDC数据集）上的性能。
主要功能包括：

1. 3D体积评估：
   - 按patient+frame分组处理所有切片
   - 将所有切片的预测结果组合成3D体积进行评估
   - 支持使用完整的3D spacing（包括x, y, z三个方向）

2. 评估指标：
   - IoU (Intersection over Union)
   - Dice系数 (DSC)
   - HD95 (95th percentile Hausdorff Distance，单位：mm)
   - NSD (Normalized Surface Distance，阈值2mm)

3. Spacing支持：
   - 从.nii.gz文件中读取spacing信息
   - 使用物理单位（mm）计算HD95和NSD指标
   - 支持3D spacing (spacing_x, spacing_y, spacing_z)

4. 数据处理：
   - 自动从COCO格式的JSON文件中读取标注
   - 支持RLE格式的mask解码
   - 自动处理图像resize（模型输入1008x1008，输出自动resize回原始尺寸）
   - 支持两种annotation格式：
     * 标准格式：categories使用"name"字段（如 "name": "right ventricle"）
     * 多文本格式：categories使用"names"字段（如 "names": ["Right ventricle cavity in cardiac MRI", ...]）
     对于多文本格式，会从每个类别的names列表中随机选择一个作为text prompt

5. 输出：
   - 控制台打印每个类别和整体的评估结果
   - 保存详细结果到文件

使用方法：
    python batch_evaluate.py --test_dir <测试集目录> --checkpoint_path <检查点路径>
    
    示例：
    python batch_evaluate.py --test_dir /home/gaoqi/sam3/dataset/ACDC/test --checkpoint_path /home/gaoqi/official_ckpt/sam3_hf/sam3.pt

参数：
    --test_dir: 测试集目录路径（必需）
    --checkpoint_path: 模型检查点路径（必需）
    --resize_size: 模型输入图像尺寸（默认1008，可选）
    --annotation_file: annotation JSON文件名（默认: image_annotations.coco.json）
                      支持 image_annotations.coco.json 或 image_annotations_multext.coco.json

注意事项：
    - 需要安装nibabel库以读取.nii.gz文件
    - 需要scipy库以计算HD95和NSD指标
    - 确保测试集目录包含ori_images文件夹（包含.nii.gz文件）

spacing读取逻辑:
ACDC: key = patient{patient_id}_frame{frame_id}
MMs2: key = {patient_id}_{frame_id}（无 "patient" 前缀）
CAMUS: 使用固定 spacing 值，不读取 spacing 文件
"""

import os
import json
import re
import argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 导入工具函数
from batch_evaluate_util import (
    compute_iou,
    compute_dice,
    compute_hd95,
    compute_nsd,
    decode_rle_mask,
    load_spacing_map
)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="SAM3模型批量评估脚本 - 3D医学图像分割评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_evaluate.py --test_dir /home/gaoqi/sam3/dataset/ACDC/test --checkpoint_path /home/gaoqi/official_ckpt/sam3_hf/sam3.pt
  python batch_evaluate.py --test_dir /home/gaoqi/sam3/dataset/MMs2/test --checkpoint_path /path/to/checkpoint.pt --resize_size 1008
        """
    )
    # /home/gaoqi/dataset/using/acdc3/test
    # /home/gaoqi/dataset/using/camus1/test
    # /home/gaoqi/dataset/using/mms2_3/test
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/home/gaoqi/dataset/using/acdc3/test",
        help="测试集目录路径"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ACDC",
        help="数据集名称（必须）。可选值: CAMUS, ACDC, MMs2"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/gaoqi/sam3/experiment/acdc_camus_weak_scribble_progress_2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt",
        help="模型检查点路径"
    )
    # /home/gaoqi/dataset/using/acdc3/test/spacing_map.json
    # /home/gaoqi/dataset/using/mms2_3/test/spacing_map.json
    parser.add_argument(
        "--spacing_file",
        type=str,
        default="/home/gaoqi/dataset/using/acdc3/test/spacing_map.json",
        help="可选：spacing 映射 JSON 文件路径（仅支持 JSON）。对于 CAMUS，可不提供，脚本将使用内置常数 spacing。"
    )

    parser.add_argument(
        "--resize_size",
        type=int,
        default=1008,
        help="模型输入图像尺寸（默认: 1008）"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="image_annotations.coco.json",
        help="annotation JSON文件名（默认: image_annotations.coco.json）。支持 image_annotations.coco.json 或 image_annotations_multext.coco.json"
    )

    args = parser.parse_args()
    
    # 配置路径
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据集路径
    test_dir = args.test_dir
    json_path = os.path.join(test_dir, args.annotation_file)
    
    # 模型配置
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = args.checkpoint_path
    resize_size = args.resize_size
    
    # 预处理检查点：处理 model 键和 detector 前缀
    # 不同来源的检查点可能有不同的键结构，需要统一处理
    print(f"预处理检查点文件: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 提取 model 键（如果存在）
    # 某些检查点格式为 {"model": {...}}，某些直接是state_dict
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    # 检查键是否包含 "detector" 前缀
    # SAM3模型的检查点键通常以 "detector." 开头
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
    
    # 加载模型架构（不通过文件路径加载检查点）
    # 先构建模型，再手动加载权重，以便更好地控制加载过程
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
    # 按照SAM3内部_load_checkpoint的逻辑处理检查点键名
    print("加载检查点到模型...")
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in state_dict.items() if "detector" in k
    }
    # 处理tracker相关的键（如果存在）
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
    
    # 构建categories字典，支持两种格式：
    # 1. 旧格式: {"id": 1, "name": "right ventricle"}
    # 2. 新格式: {"id": 1, "names": ["Right ventricle cavity in cardiac MRI", ...]}
    categories_dict = {}
    categories_names_dict = {}  # 存储新格式的names列表（如果存在）
    use_multext_format = False
    
    for cat in coco_data['categories']:
        cat_id = cat['id']
        if 'names' in cat and isinstance(cat['names'], list):
            # 新格式：有names列表
            use_multext_format = True
            categories_names_dict[cat_id] = cat['names']
            # 默认使用第一个name作为显示名称（用于日志输出）
            categories_dict[cat_id] = cat['names'][0]
        elif 'name' in cat:
            # 旧格式：只有name字段
            categories_dict[cat_id] = cat['name']
        else:
            raise ValueError(f"Category {cat_id} must have either 'name' or 'names' field")
    
    if use_multext_format:
        print(f"检测到多文本格式annotation文件，将从每个类别的names列表中随机选择prompt")
    else:
        print(f"使用标准格式annotation文件")
    
    # 按图像ID组织标注
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # 按patient+frame分组所有图像（用于3D评估）
    # 不同数据集的文件命名格式不同，需要统一解析
    def extract_patient_frame(img_file):
        """从文件名提取patient_id、frame_id/view/phase和slice_idx
        
        支持三种数据集格式：
        - ACDC: patient{数字}_frame{数字}_slice{数字}.png
        - MMs2: {patient_id}_{view}_{phase}_slice{数字}.png (view: SA/LA, phase: ED/ES)
        - CAMUS: patient{数字}_{view}_{phase}_slice{数字}.png (view: 2CH/4CH等, phase: ED/ES)
        
        Returns:
            (patient_id, frame_id, slice_idx, view) 或 (None, None, None, None) 如果无法匹配
        """
        basename = os.path.basename(img_file)
        
        # 尝试匹配ACDC格式: patient{数字}_frame{数字}_slice{数字}.png
        match_acdc = re.match(r'patient(\d+)_frame(\d+)_slice(\d+)\.png', basename)
        if match_acdc:
            patient_id = match_acdc.group(1)
            frame_id = match_acdc.group(2)
            slice_idx = int(match_acdc.group(3))
            return patient_id, frame_id, slice_idx, None  # ACDC格式没有view
        
        # 尝试匹配MMs2格式: {patient_id}_{view}_{phase}_slice{数字}.png
        match_mms2 = re.match(r'(\d+)_(SA|LA)_(ED|ES)_slice(\d+)\.png', basename)
        if match_mms2:
            patient_id = match_mms2.group(1)
            view = match_mms2.group(2)  # SA或LA
            phase = match_mms2.group(3)  # ED或ES
            slice_idx = int(match_mms2.group(4))
            # 使用view+phase作为frame_id的替代
            frame_id = f"{view}_{phase}"
            return patient_id, frame_id, slice_idx, view
        
        # 尝试匹配CAMUS格式: patient{数字}_{view}_{phase}.png 或 patient{数字}_{view}_{phase}_slice{数字}.png
        # view可能是2CH、4CH等
        # CAMUS数据集通常是单切片的，所以可能没有slice后缀
        match_camus = re.match(r'patient(\d+)_([A-Z0-9]+)_(ED|ES)(?:_slice(\d+))?\.png', basename)
        if match_camus:
            patient_id = match_camus.group(1)
            view = match_camus.group(2)  # 2CH、4CH等
            phase = match_camus.group(3)  # ED或ES
            # 如果有slice索引，使用它；否则设置为0（单切片情况）
            slice_idx_str = match_camus.group(4)
            slice_idx = int(slice_idx_str) if slice_idx_str else 0
            # 使用view+phase作为frame_id的替代
            frame_id = f"{view}_{phase}"
            return patient_id, frame_id, slice_idx, view
        
        return None, None, None, None
    
    # 按patient+frame分组
    images_by_volume = defaultdict(list)  # key: (patient_id, frame_id), value: list of (img_id, slice_idx, view)
    for img_id, img_info in images_dict.items():
        patient_id, frame_id, slice_idx, view = extract_patient_frame(img_info['file_name'])
        if patient_id and frame_id:
            images_by_volume[(patient_id, frame_id)].append((img_id, slice_idx, view))
    
    # 对于MMs2数据集，过滤掉LA视图，只保留SA视图
    # MMs2数据集包含SA（短轴）和LA（长轴）两种视图，通常只评估SA视图
    # 检查是否有view信息（MMs2或CAMUS格式）
    has_view_info = False
    is_mms2_format = False  # 用于区分MMs2和CAMUS
    for img_list in images_by_volume.values():
        if img_list and len(img_list[0]) == 3 and img_list[0][2] is not None:
            has_view_info = True
            view = img_list[0][2]
            # MMs2的view是SA或LA，CAMUS的view是2CH、4CH等
            if view in ["SA", "LA"]:
                is_mms2_format = True
            break
    
    if has_view_info and is_mms2_format:
        # 检测到MMs2格式（有view信息且view是SA或LA），过滤LA视图
        filtered_images_by_volume = defaultdict(list)
        for key, img_list in images_by_volume.items():
            # 检查该volume的第一个图像的view（img_list[0][2]是view）
            if img_list and len(img_list[0]) == 3 and img_list[0][2] == "SA":
                # 只保留SA视图的volume
                filtered_images_by_volume[key] = img_list
        images_by_volume = filtered_images_by_volume
        print(f"Detected MMs2 dataset. Filtered to SA view only: {len(images_by_volume)} volumes remaining")
    elif has_view_info and not is_mms2_format:
        # 检测到CAMUS格式（有view信息但view不是SA或LA），不过滤
        # CAMUS数据集包含2CH、4CH等视图，需要全部评估
        print(f"Detected CAMUS dataset. Processing all views: {len(images_by_volume)} volumes")
    
    # 对每个volume，按slice_idx排序
    for key in images_by_volume:
        images_by_volume[key].sort(key=lambda x: x[1])  # x[1]是slice_idx
    
    # 统计结果
    category_ious = defaultdict(list)
    category_dices = defaultdict(list)
    category_hd95s = defaultdict(list)
    category_nsds = defaultdict(list)
    all_ious = []
    all_dices = []
    all_hd95s = []
    all_nsds = []
    
    # 记录每个patient的详细结果
    per_patient = {}

    # 使用用户提供的数据集名称决定 spacing 处理逻辑
    dataset_name = args.dataset_name.strip()
    dataset_key = dataset_name.lower()
    allowed = {"camus", "acdc", "mms2"}
    if dataset_key not in allowed:
        print(f"Error: unknown dataset_name '{dataset_name}'. Allowed: CAMUS, ACDC, MMs2")
        return
    is_camus = (dataset_key == 'camus')
    is_mms2 = (dataset_key == 'mms2')
    is_acdc = (dataset_key == 'acdc')
    spacing_map = {}
    if is_camus:
        print("Detected CAMUS dataset — will use constant spacing for all volumes.")
    else:
        if args.spacing_file is None:
            print("Error: spacing_file is required for non-CAMUS datasets.")
            return
        print(f"Loading spacing map from: {args.spacing_file} (JSON)")
        spacing_map = load_spacing_map(args.spacing_file)
        print(f"Loaded {len(spacing_map)} spacing entries")
    
    # 遍历所有volume（patient+frame组合）
    print("Starting 3D volume evaluation ...")
    volume_keys = sorted(images_by_volume.keys())
    
    for patient_id, frame_id in tqdm(volume_keys, desc="Processing volumes"):
        # 获取该volume的所有切片
        slice_list = images_by_volume[(patient_id, frame_id)]
        if len(slice_list) == 0:
            continue
        
        # 检查第一个图像以确定格式（用于构造spacing map的key）
        first_img_id, first_slice_idx, first_view = images_by_volume[(patient_id, frame_id)][0]
        
        # 获取3D spacing
        # CAMUS数据集使用固定的spacing值；ACDC和MMs2数据集从spacing_map中读取
        if is_camus:
            # CAMUS数据集的固定spacing值（单位：mm）
            spacing_3d = (0.30799999833106995, 0.30799999833106995, 1.0)
        else:
            # 构造spacing map的key（格式与JSON文件中的键一致）
            # spacing文件的key是从.nii.gz文件名去掉后缀得到的
            if is_acdc:
                # ACDC格式: .nii.gz文件名是 patient{patient_id}_frame{frame_id}.nii.gz
                # 所以key是: patient{patient_id}_frame{frame_id}
                key = f"patient{patient_id}_frame{frame_id}"
            elif is_mms2:
                # MMs2格式: .nii.gz文件名是 {patient_id}_{view}_{phase}.nii.gz（没有"patient"前缀）
                # 所以key是: {patient_id}_{view}_{phase}，即 {patient_id}_{frame_id}
                key = f"{patient_id}_{frame_id}"
            else:
                # 其他情况（理论上不会执行到这里，因为只有ACDC、MMs2、CAMUS三种）
                # 如果first_view为None，假设是ACDC格式
                if first_view is None:
                    key = f"patient{patient_id}_frame{frame_id}"
                else:
                    # 如果有view信息，假设是CAMUS格式（但CAMUS不使用spacing文件）
                    key = f"patient{patient_id}_{frame_id}"

            if key in spacing_map:
                spacing_3d = spacing_map[key]
            else:
                print(f"Warning: spacing not found for key '{key}' in spacing JSON. Skipping volume {patient_id}_{frame_id}.")
                # 打印一些调试信息，帮助用户排查问题
                if len(spacing_map) > 0:
                    sample_keys = list(spacing_map.keys())[:3]
                    print(f"  Available keys in spacing_map (first 3): {sample_keys}")
                continue
        
        # 获取该volume中所有类别（从所有切片的标注中收集）
        volume_categories = set()
        for img_id, _, _ in slice_list:
            for ann in annotations_by_image[img_id]:
                volume_categories.add(ann['category_id'])
        
        if len(volume_categories) == 0:
            continue
        
        # 记录该volume的每个类别结果
        volume_per_class = {}
        volume_overall_ious = []
        volume_overall_dices = []
        volume_overall_hd95s = []
        volume_overall_nsds = []
        
        # 对每个类别进行3D评估
        for category_id in volume_categories:
            category_name = categories_dict[category_id]
            
            # 为当前类别选择text prompt（如果是多文本格式，随机选择一个；否则使用标准名称）
            # 整个volume的该类别都使用同一个prompt，保持一致性
            if category_id in categories_names_dict:
                # 从names列表中随机选择一个作为prompt
                text_prompt = np.random.choice(categories_names_dict[category_id]).lower()
            else:
                # 使用标准的category_name
                text_prompt = category_name.lower()
            
            # 收集该类别在所有切片上的预测结果
            pred_masks_2d = []  # 存储每个切片的2D预测mask
            gt_masks_2d = []    # 存储每个切片的2D GT mask
            slice_indices = []  # 存储切片索引
            
            # 处理每个切片
            for img_id, slice_idx, view in slice_list:
                img_info = images_dict[img_id]
                img_file = img_info['file_name']
                img_path = os.path.join(test_dir, img_file)
                
                if not os.path.exists(img_path):
                    continue
                
                # 加载图像
                image = Image.open(img_path)
                img_height, img_width = img_info['height'], img_info['width']
                
                # 获取该切片的标注
                slice_annotations = [ann for ann in annotations_by_image[img_id] 
                                   if ann['category_id'] == category_id]
                
                if len(slice_annotations) == 0:
                    # 该切片没有该类别的标注，创建空的mask
                    gt_mask_2d = np.zeros((img_height, img_width), dtype=bool)
                else:
                    # 使用第一个标注（假设每个切片每个类别只有一个标注）
                    ann = slice_annotations[0]
                    segm = ann['segmentation']
                    gt_mask_2d = decode_rle_mask(segm, img_height, img_width)
                
                # 使用SAM3模型进行推理
                # 设置图像并重置所有prompt
                inference_state = processor.set_image(image)
                processor.reset_all_prompts(inference_state)
                # 使用预先选择的text prompt进行分割（整个volume的该类别使用同一个prompt）
                inference_state = processor.set_text_prompt(
                    state=inference_state, 
                    prompt=text_prompt
                )
                
                # 获取推理结果
                masks = inference_state["masks"]
                scores = inference_state["scores"]
                
                if len(masks) == 0:
                    # 没有检测到任何mask，创建空的预测mask
                    pred_mask_2d = np.zeros((img_height, img_width), dtype=bool)
                else:
                    # 选择score最高的mask作为预测结果
                    best_idx = scores.argmax().item()
                    pred_mask_tensor = masks[best_idx]
                    
                    # mask的形状可能是 (1, H, W) 或 (H, W)，需要统一处理
                    if pred_mask_tensor.dim() == 3:
                        pred_mask_tensor = pred_mask_tensor.squeeze(0)
                    
                    # 转换为numpy数组
                    pred_mask_2d = pred_mask_tensor.cpu().numpy().astype(bool)
                    
                    # 如果尺寸不匹配，需要resize到原始图像尺寸
                    # 模型输入尺寸是resize_size（默认1008），输出需要resize回原始尺寸
                    if pred_mask_2d.shape != gt_mask_2d.shape:
                        pred_mask_pil = Image.fromarray(pred_mask_2d.astype(np.uint8) * 255)
                        pred_mask_pil = pred_mask_pil.resize((img_width, img_height), Image.NEAREST)
                        pred_mask_2d = np.array(pred_mask_pil) > 0
                
                # 保存该切片的预测和GT
                pred_masks_2d.append(pred_mask_2d)
                gt_masks_2d.append(gt_mask_2d)
                slice_indices.append(slice_idx)
            
            if len(pred_masks_2d) == 0:
                continue
            
            # 将所有切片组合成3D体积进行评估
            # 确保所有mask尺寸一致
            h, w = pred_masks_2d[0].shape
            num_slices = len(pred_masks_2d)
            
            # 创建3D mask，按slice_idx排序
            # 由于slice_list已经按slice_idx排序，我们只需要按顺序组合
            pred_mask_3d = np.zeros((num_slices, h, w), dtype=bool)
            gt_mask_3d = np.zeros((num_slices, h, w), dtype=bool)
            
            for i in range(num_slices):
                pred_mask_3d[i] = pred_masks_2d[i]
                gt_mask_3d[i] = gt_masks_2d[i]
            
            # 进行3D评估：计算IoU和Dice系数
            iou = compute_iou(pred_mask_3d, gt_mask_3d)
            dice = compute_dice(pred_mask_3d, gt_mask_3d)
            
            # 计算HD95（使用3D spacing，单位为mm）
            # HD95是95百分位Hausdorff距离，衡量边界匹配程度
            try:
                hd95 = compute_hd95(pred_mask_3d, gt_mask_3d, spacing=spacing_3d)
            except Exception as e:
                print(f"Error computing HD95 for {category_name} in patient{patient_id}_frame{frame_id}: {e}")
                hd95 = float('nan')
            
            # 计算NSD（Normalized Surface Distance，阈值2mm）
            # NSD衡量在2mm阈值范围内的表面点比例
            try:
                nsd = compute_nsd(pred_mask_3d, gt_mask_3d, spacing=spacing_3d, threshold_mm=2.0)
            except Exception as e:
                print(f"Error computing NSD for {category_name} in patient{patient_id}_frame{frame_id}: {e}")
                nsd = float('nan')
            
            # 记录全局结果
            category_ious[category_name].append(iou)
            category_dices[category_name].append(dice)
            category_hd95s[category_name].append(hd95)
            category_nsds[category_name].append(nsd)
            all_ious.append(iou)
            all_dices.append(dice)
            all_hd95s.append(hd95)
            all_nsds.append(nsd)
            
            # 记录该volume的类别结果
            volume_per_class[category_name] = {
                "dice": float(dice),
                "iou": float(iou),
                "hd95": float(hd95) if not (np.isnan(hd95) or np.isinf(hd95)) else None,
                "nsd": float(nsd) if not np.isnan(nsd) else None,
                "pred_sum": int(pred_mask_3d.sum()),
                "gt_sum": int(gt_mask_3d.sum())
            }
            
            # 记录该volume的overall结果（用于计算平均值）
            volume_overall_ious.append(iou)
            volume_overall_dices.append(dice)
            if not (np.isnan(hd95) or np.isinf(hd95)):
                volume_overall_hd95s.append(hd95)
            if not np.isnan(nsd):
                volume_overall_nsds.append(nsd)
        
        # 计算该volume的overall指标（所有类别的平均值）
        volume_overall_dice = float(np.mean(volume_overall_dices)) if len(volume_overall_dices) > 0 else float('nan')
        volume_overall_iou = float(np.mean(volume_overall_ious)) if len(volume_overall_ious) > 0 else float('nan')
        volume_overall_hd95 = float(np.nanmean(volume_overall_hd95s)) if len(volume_overall_hd95s) > 0 else float('nan')
        volume_overall_nsd = float(np.nanmean(volume_overall_nsds)) if len(volume_overall_nsds) > 0 else float('nan')
        
        # 记录该patient的结果
        # 根据格式生成patient_name
        if first_view is None:
            # ACDC格式: frame_id是 "01" 等
            patient_name = f"patient{patient_id}_frame{frame_id}"
        elif first_view in ["SA", "LA"]:
            # MMs2格式: frame_id是 "SA_ED" 或 "LA_ES" 等
            patient_name = f"{patient_id}_{frame_id}"
        else:
            # CAMUS格式: frame_id是 "2CH_ED" 或 "4CH_ES" 等
            patient_name = f"patient{patient_id}_{frame_id}"
        per_patient[patient_name] = {
            "per_class": {str(cat): volume_per_class[cat] for cat in volume_per_class},
            "overall": {
                "dice": volume_overall_dice,
                "iou": volume_overall_iou,
                "hd95": volume_overall_hd95 if not (np.isnan(volume_overall_hd95) or np.isinf(volume_overall_hd95)) else None,
                "nsd": volume_overall_nsd if not np.isnan(volume_overall_nsd) else None
            }
        }
        
        # 打印该patient的结果
        hd95_str = f", HD95={volume_overall_hd95:.4f}" if not (np.isnan(volume_overall_hd95) or np.isinf(volume_overall_hd95)) else ", HD95=N/A"
        nsd_str = f", NSD={volume_overall_nsd:.4f}" if not np.isnan(volume_overall_nsd) else ", NSD=N/A"
        print(f"{patient_name}: overall Dice={volume_overall_dice:.4f}, IoU={volume_overall_iou:.4f}{hd95_str}{nsd_str}")
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nPer-category metrics:")
    print("-"*100)
    print(f"{'Category':<20} {'IoU':<12} {'Dice':<12} {'HD95(mm)':<12} {'NSD(2mm)':<12} {'Count':<10}")
    print("-"*100)
    
    for category_name in sorted(category_ious.keys()):
        ious = category_ious[category_name]
        dices = category_dices[category_name]
        hd95s = category_hd95s[category_name]
        nsds = category_nsds[category_name]
        mean_iou = np.mean(ious) if len(ious) > 0 else float('nan')
        mean_dice = np.mean(dices) if len(dices) > 0 else float('nan')
        mean_hd95 = float(np.nanmean(hd95s)) if len(hd95s) > 0 else float('nan')
        mean_nsd = float(np.nanmean(nsds)) if len(nsds) > 0 else float('nan')
        count = len(ious)
        print(f"{category_name:<20} {mean_iou:<12.4f} {mean_dice:<12.4f} {mean_hd95:<12.2f} {mean_nsd:<12.4f} {count:<10}")
    
    print("-"*100)
    overall_mean_iou = np.mean(all_ious) if len(all_ious) > 0 else float('nan')
    overall_mean_dice = np.mean(all_dices) if len(all_dices) > 0 else float('nan')
    overall_mean_hd95 = float(np.nanmean(all_hd95s)) if len(all_hd95s) > 0 else float('nan')
    overall_mean_nsd = float(np.nanmean(all_nsds)) if len(all_nsds) > 0 else float('nan')
    print(f"{'Overall':<20} {overall_mean_iou:<12.4f} {overall_mean_dice:<12.4f} {overall_mean_hd95:<12.2f} {overall_mean_nsd:<12.4f} {len(all_ious):<10}")
    print("="*100)
    
    # 保存结果到JSON文件
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算每个类别的汇总统计（用于JSON输出）
    per_class_summary = {}
    for category_name in sorted(category_ious.keys()):
        ious = category_ious[category_name]
        dices = category_dices[category_name]
        hd95s = category_hd95s[category_name]
        nsds = category_nsds[category_name]
        
        # 过滤无效的HD95值（nan和inf）
        valid_hd95s = [h for h in hd95s if not (np.isnan(h) or np.isinf(h))]
        valid_nsds = [n for n in nsds if not np.isnan(n)]
        
        per_class_summary[category_name] = {
            "dice_mean": float(np.mean(dices)) if len(dices) > 0 else float('nan'),
            "iou_mean": float(np.mean(ious)) if len(ious) > 0 else float('nan'),
            "hd95_mean": float(np.nanmean(valid_hd95s)) if len(valid_hd95s) > 0 else float('nan'),
            "nsd_mean": float(np.nanmean(valid_nsds)) if len(valid_nsds) > 0 else float('nan'),
            "count": len(ious)
        }
    
    # 计算overall汇总统计
    valid_overall_hd95s = [h for h in all_hd95s if not (np.isnan(h) or np.isinf(h))]
    valid_overall_nsds = [n for n in all_nsds if not np.isnan(n)]
    
    overall_summary = {
        "dice_mean": float(np.mean(all_dices)) if len(all_dices) > 0 else float('nan'),
        "iou_mean": float(np.mean(all_ious)) if len(all_ious) > 0 else float('nan'),
        "hd95_mean": float(np.nanmean(valid_overall_hd95s)) if len(valid_overall_hd95s) > 0 else float('nan'),
        "nsd_mean": float(np.nanmean(valid_overall_nsds)) if len(valid_overall_nsds) > 0 else float('nan')
    }
    
    # 保存JSON格式结果（包含per_patient信息）
    output_file_json = os.path.join(output_dir, f"evaluation_results_{dataset_name.lower()}.json")
    
    # 定义JSON编码器，处理NaN和None值
    def json_encoder(obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None
        return obj
    
    # 递归清理字典中的NaN值
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        elif isinstance(d, float):
            if np.isnan(d) or np.isinf(d):
                return None
        return d
    
    output_dict = {
        "per_class": clean_dict(per_class_summary),
        "overall": clean_dict(overall_summary),
        "per_patient": clean_dict(per_patient),
        "n_cases": len(per_patient)
    }
    
    with open(output_file_json, 'w') as f:
        json.dump(output_dict, f, indent=2, default=json_encoder)
    
    print(f"\nResults saved to: {output_file_json}")
    print(f"  - Per-class summary: {len(per_class_summary)} categories")
    print(f"  - Per-patient results: {len(per_patient)} cases")


if __name__ == "__main__":
    main()

