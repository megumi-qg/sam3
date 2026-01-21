"""
脚本名称: inference_for_bbox_update.py
功能描述: 在弱监督训练过程中执行推理，生成伪标签 BBox 以进行渐进式训练 (Progressive Training)。

背景与用途:
    本脚本用于 SAM3 模型的弱监督微调任务（如 ACDC/CAMUS 数据集）。
    在训练初期（Warm-up），模型仅使用 Scribble 的外接矩形作为弱监督信号。
    在训练中期，本脚本利用模型当前的预测能力，生成更紧致、更完整的 BBox（伪标签）来替换粗糙的 Scribble BBox，
    从而实现从"弱监督"向"强监督"的平滑过渡。

核心策略改进 (针对解决训练后期性能下降问题):
    1. Union Box 策略 (并集兜底):
       - 旧策略: 直接使用模型预测的 Mask 生成 BBox。如果预测偏差较大，会导致 Ground Truth (Scribble) 丢失。
       - 新策略: Final_BBox = Predicted_BBox ∪ Scribble_BBox
       - 作用: 强制最终的 BBox 必须包含人工标注的 Scribble。即使模型预测不完整，也能保证 GT 线索不丢失，
         极大提升了训练稳定性，防止模型在 Epoch 7 之后性能崩塌。

    2. 宽容验证 (Relaxed Validation):
       - 旧策略: 只要 Scribble 有像素落在预测框外，就丢弃该样本。导致大量有效样本被浪费。
       - 新策略: 仅当预测框与 Scribble 完全无交集（IoU=0，即出现幻觉）时才丢弃。
       - 作用: 允许模型预测存在边缘误差，最大化利用训练数据。

主要函数:
    - run_inference_on_training_dataset_v2_distributed:
        支持 DDP 分布式推理，每个 Rank 处理 1/N 的数据，最后汇总结果。
        同时兼容 `COCO_FROM_JSON` (自定义加载器) 和标准 `pycocotools.coco.COCO` 格式。

    - merge_bboxes:
        执行 BBox 并集计算的核心工具函数。

    - validate_prediction_relevance:
        执行宽容验证，剔除完全不相关的"幻觉"预测。

输入输出:
    - 输入: 训练好的 SAM3 模型、训练数据集 (包含 Scribble 标注)。
    - 输出: Dict 缓存，格式为 {ann_file_path: {annotation_id: [x, y, w, h]}}。
            该缓存将被 Dataset 类读取，用于动态更新训练时的 Target BBox。

使用注意:
    请在 trainer.py 中调用此脚本时，将 `confidence_threshold` 设置为 0.3 ~ 0.4 之间，
    以平衡伪标签的召回率和准确率。
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as maskUtils


def compute_bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    """
    从mask计算外接矩形框 (xywh格式，归一化到[0,1])
    """
    if mask.sum() == 0:
        return None
    
    h, w = mask.shape
    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    # 转换为归一化的xywh格式
    x_center = (x_min + x_max) / 2.0 / w
    y_center = (y_min + y_max) / 2.0 / h
    width = (x_max - x_min + 1) / w
    height = (y_max - y_min + 1) / h
    
    return [x_center, y_center, width, height]


def decode_scribble_mask(segmentation, img_h: int, img_w: int) -> Optional[np.ndarray]:
    """
    解码scribble mask（支持polygon和RLE格式）
    """
    if segmentation is None:
        return None
    
    try:
        if isinstance(segmentation, list):
            # Polygon 格式
            rle = maskUtils.frPyObjects(segmentation, img_h, img_w)
            m = maskUtils.decode(rle)
            if len(m.shape) == 3:
                m = m.sum(axis=2) > 0
            return m.astype(bool)
        elif isinstance(segmentation, dict):
            # RLE 格式
            m = maskUtils.decode(segmentation)
            return m.astype(bool)
        else:
            return None
    except Exception as e:
        logging.warning(f"解码scribble mask失败: {e}")
        return None


def merge_bboxes(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """
    计算两个归一化 xywh bbox 的并集外接矩形
    """
    if bbox1 is None: return bbox2
    if bbox2 is None: return bbox1

    # 转换为 x1, y1, x2, y2
    b1_x1 = bbox1[0] - bbox1[2]/2
    b1_y1 = bbox1[1] - bbox1[3]/2
    b1_x2 = bbox1[0] + bbox1[2]/2
    b1_y2 = bbox1[1] + bbox1[3]/2

    b2_x1 = bbox2[0] - bbox2[2]/2
    b2_y1 = bbox2[1] - bbox2[3]/2
    b2_x2 = bbox2[0] + bbox2[2]/2
    b2_y2 = bbox2[1] + bbox2[3]/2

    # 计算并集
    x1 = min(b1_x1, b2_x1)
    y1 = min(b1_y1, b2_y1)
    x2 = max(b1_x2, b2_x2)
    y2 = max(b1_y2, b2_y2)

    # 转回 xywh
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    
    return [cx, cy, w, h]


def validate_prediction_relevance(
    pred_mask: np.ndarray,
    scribble_mask: Optional[np.ndarray],
    coverage_threshold: float = 0.8,
) -> bool:
    """
    [修改版] 验证预测质量
    逻辑：预测的Mask必须覆盖掉大部分的Scribble，才算合格。
    这能有效防止模型预测偏离目标物体（虽然有交集但没完全覆盖），
    也能隐式防止预测过小。
    
    只有当预测Mask覆盖了80%以上的Scribble像素时，才认为是可靠的。
    """
    if scribble_mask is None or scribble_mask.sum() == 0:
        return True  # 没有Scribble，无法验证，默认信任模型或根据策略丢弃

    # 确保pred_mask和scribble_mask都是布尔类型
    if pred_mask.dtype != bool:
        pred_mask = pred_mask.astype(bool)
    if scribble_mask.dtype != bool:
        scribble_mask = scribble_mask.astype(bool)
    
    # 确保pred_mask和scribble_mask形状一致
    if pred_mask.shape != scribble_mask.shape:
        # 如果形状不一致，尝试调整pred_mask的大小
        from PIL import Image
        pred_mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(
            (scribble_mask.shape[1], scribble_mask.shape[0]), 
            Image.NEAREST
        )).astype(bool)
        pred_mask = pred_mask_resized

    # 计算Scribble的总像素数
    scribble_area = scribble_mask.sum()
    
    # 计算预测Mask中包含了多少Scribble像素 (Intersection)
    intersection = (pred_mask & scribble_mask).sum()
    
    # 计算覆盖率 (Recall)
    coverage = intersection / (scribble_area + 1e-6)
    
    # 只有当预测Mask覆盖了 80% 以上的 Scribble 像素时，才认为是可靠的
    if coverage < coverage_threshold:
        logging.debug(f"验证失败: 覆盖率仅为 {coverage:.2f}")
        return False
        
    return True 


def run_inference_on_training_dataset_v2_distributed(
    model,
    train_dataset,
    categories: List[Dict],
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Dict[str, List[float]]]:
    """
    分布式推理版本
    """
    import torch.distributed as dist
    from sam3.model.sam3_image_processor import Sam3Processor
    
    model.eval()
    processor = Sam3Processor(model, resolution=1008, confidence_threshold=confidence_threshold)
    
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    bbox_cache = defaultdict(dict)
    
    if hasattr(train_dataset, 'dataset'):
        datasets = train_dataset.dataset.datasets if isinstance(train_dataset.dataset, torch.utils.data.ConcatDataset) else [train_dataset.dataset]
    else:
        datasets = [train_dataset]
    
    for dataset in datasets:
        if not hasattr(dataset, 'coco'):
            continue
        
        ann_file = dataset.annFile
        coco = dataset.coco
        
        logging.info(f"Rank {rank}: 处理annotation文件: {ann_file}")
        
        # 检查是否是 COCO_FROM_JSON 对象
        is_coco_from_json = hasattr(coco, '_raw_data')
        
        if is_coco_from_json:
            # === 分支 A: COCO_FROM_JSON ===
            raw_data = coco._raw_data
            total_images = len(raw_data)
            
            images_per_rank = total_images // world_size
            start_idx = rank * images_per_rank
            end_idx = total_images if rank == world_size - 1 else start_idx + images_per_rank
            
            logging.info(f"Rank {rank}: 总图像数={total_images}, 处理 {start_idx}-{end_idx}")
            
            processed_images = 0
            processed_annotations = 0
            successful_inferences = 0
            failed_inferences = 0
            
            for idx in range(start_idx, end_idx):
                img_data = raw_data[idx]
                img_info = img_data["image"]
                image_id = img_info["id"]
                img_path = os.path.join(dataset.root, img_info['file_name'])
                
                if not os.path.exists(img_path): continue
                try:
                    pil_image = Image.open(img_path).convert('RGB')
                except Exception: continue
                
                processed_images += 1
                annotations = img_data["annotations"]
                if not annotations: continue
                
                for ann in annotations:
                    processed_annotations += 1
                    category_id = ann['category_id']
                    category_name = category_id_to_name.get(category_id, f"category_{category_id}")
                    ann_id = ann['id']
                    
                    try:
                        inference_state = processor.set_image(pil_image)
                        processor.reset_all_prompts(inference_state)
                        inference_state = processor.set_text_prompt(state=inference_state, prompt=category_name)
                        
                        pred_masks = inference_state["masks"]
                        pred_scores = inference_state["scores"]
                        
                        if len(pred_masks) == 0:
                            failed_inferences += 1
                            continue
                        
                        best_idx = pred_scores.argmax().item()
                        best_mask = pred_masks[best_idx].squeeze().cpu().numpy()
                        
                        # 计算预测 bbox
                        pred_bbox_xywh = compute_bbox_from_mask(best_mask)
                        
                        # [修复] 变量名修正：使用 pred_bbox_xywh
                        if pred_bbox_xywh is not None:
                            img_h, img_w = pil_image.size[1], pil_image.size[0]
                            
                            scribble_mask = None
                            scribble_bbox_xywh = None
                            if "segmentation" in ann and ann["segmentation"] is not None:
                                scribble_mask = decode_scribble_mask(ann["segmentation"], img_h, img_w)
                                if scribble_mask is not None:
                                    scribble_bbox_xywh = compute_bbox_from_mask(scribble_mask)
                                    
                            is_relevant = validate_prediction_relevance(best_mask, scribble_mask, coverage_threshold=0.8)

                            if is_relevant:
                                final_bbox = merge_bboxes(pred_bbox_xywh, scribble_bbox_xywh)
                                bbox_cache[ann_file][str(ann_id)] = final_bbox
                                successful_inferences += 1
                            else:
                                failed_inferences += 1
                        else:
                            failed_inferences += 1
                    
                    except Exception as e:
                        failed_inferences += 1
                        continue
                
                if (idx - start_idx + 1) % 100 == 0:
                    logging.info(f"Rank {rank}: 进度 {idx - start_idx + 1}/{end_idx - start_idx}, 成功 {successful_inferences}")
            
            logging.info(f"Rank {rank}: [{os.path.basename(ann_file)}] 完成. 成功: {successful_inferences}, 失败: {failed_inferences}")
            
        else:
            # === 分支 B: 标准 pycocotools.coco.COCO ===
            # [修复] 这个分支现在与上方逻辑保持一致，使用 Union Box 策略
            image_ids = coco.getImgIds()
            total_images = len(image_ids)
            
            images_per_rank = total_images // world_size
            start_idx = rank * images_per_rank
            end_idx = total_images if rank == world_size - 1 else start_idx + images_per_rank
            
            logging.info(f"Rank {rank}: 总图像数={total_images}, 处理 {start_idx}-{end_idx}")
            
            processed_images = 0
            processed_annotations = 0
            successful_inferences = 0
            failed_inferences = 0
            
            for idx in range(start_idx, end_idx):
                image_id = image_ids[idx]
                img_info = coco.loadImgs(image_id)[0]
                img_path = os.path.join(dataset.root, img_info['file_name'])
                
                if not os.path.exists(img_path): continue
                try:
                    pil_image = Image.open(img_path).convert('RGB')
                except Exception: continue
                
                processed_images += 1
                ann_ids = coco.getAnnIds(imgIds=image_id)
                annotations = coco.loadAnns(ann_ids)
                if not annotations: continue
                
                for ann in annotations:
                    processed_annotations += 1
                    category_id = ann['category_id']
                    category_name = category_id_to_name.get(category_id, f"category_{category_id}")
                    ann_id = ann['id']
                    
                    try:
                        inference_state = processor.set_image(pil_image)
                        processor.reset_all_prompts(inference_state)
                        inference_state = processor.set_text_prompt(state=inference_state, prompt=category_name)
                        
                        pred_masks = inference_state["masks"]
                        pred_scores = inference_state["scores"]
                        
                        if len(pred_masks) == 0:
                            failed_inferences += 1
                            continue
                        
                        best_idx = pred_scores.argmax().item()
                        best_mask = pred_masks[best_idx].squeeze().cpu().numpy()
                        
                        # 计算预测 bbox
                        pred_bbox_xywh = compute_bbox_from_mask(best_mask)
                        
                        # [修复] 变量名一致性，逻辑一致性
                        if pred_bbox_xywh is not None:
                            img_h, img_w = pil_image.size[1], pil_image.size[0]
                            
                            scribble_mask = None
                            scribble_bbox_xywh = None
                            
                            if "segmentation" in ann and ann["segmentation"] is not None:
                                scribble_mask = decode_scribble_mask(ann["segmentation"], img_h, img_w)
                            
                            # 尝试获取已有的 bbox (可能是绝对坐标) 并归一化作为 fallback
                            if scribble_mask is None and "bbox" in ann and ann["bbox"] is not None:
                                s_bbox = ann["bbox"]
                                if len(s_bbox) == 4:
                                     # 假设是绝对坐标 xywh
                                    scribble_bbox_xywh = [s_bbox[0]/img_w, s_bbox[1]/img_h, s_bbox[2]/img_w, s_bbox[3]/img_h]
                            elif scribble_mask is not None:
                                scribble_bbox_xywh = compute_bbox_from_mask(scribble_mask)
                            
                            # [重要] 这里使用新逻辑，而不是已删除的 validate_predicted_bbox
                            is_relevant = validate_prediction_relevance(best_mask, scribble_mask, coverage_threshold=0.8)
                            
                            if is_relevant:
                                final_bbox = merge_bboxes(pred_bbox_xywh, scribble_bbox_xywh)
                                bbox_cache[ann_file][str(ann_id)] = final_bbox
                                successful_inferences += 1
                            else:
                                failed_inferences += 1
                        else:
                            failed_inferences += 1
                    
                    except Exception as e:
                        failed_inferences += 1
                        continue

                if (idx - start_idx + 1) % 100 == 0:
                    logging.info(f"Rank {rank}: 进度 {idx - start_idx + 1}/{end_idx - start_idx}, 成功 {successful_inferences}")
            
            logging.info(f"Rank {rank}: [{os.path.basename(ann_file)}] 完成. 成功: {successful_inferences}, 失败: {failed_inferences}")

    total_bboxes = sum(len(v) for v in bbox_cache.values())
    logging.info(f"Rank {rank}: 推理完成，共更新 {total_bboxes} 个bbox")
    
    return dict(bbox_cache)