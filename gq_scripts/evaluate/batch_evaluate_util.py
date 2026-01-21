"""
工具函数模块 - 用于batch_evaluate.py

包含评估指标计算、数据加载等工具函数：
- IoU、Dice、HD95、NSD等评估指标计算
- COCO格式mask解码
- Spacing映射文件加载
"""

import json
import os
import numpy as np
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

# scipy库可用性标志（用于HD95和NSD计算）
SCIPY_AVAILABLE = True


def compute_iou(pred_mask, gt_mask):
    """计算IoU (Intersection over Union)
    
    Args:
        pred_mask: 预测的二值mask，可以是2D或3D
        gt_mask: 真实标签的二值mask，形状与pred_mask相同
    
    Returns:
        IoU值，范围[0, 1]
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """计算Dice系数（Dice Similarity Coefficient, DSC）
    
    Args:
        pred_mask: 预测的二值mask，可以是2D或3D
        gt_mask: 真实标签的二值mask，形状与pred_mask相同
    
    Returns:
        Dice系数，范围[0, 1]
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    if pred_area + gt_area == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / (pred_area + gt_area)


def compute_hd95(pred_mask, gt_mask, spacing=None):
    """计算95th percentile Hausdorff Distance (HD95)，支持2D和3D。
    
    Args:
        pred_mask: 预测的二值mask，可以是2D (H, W) 或 3D (D, H, W)
        gt_mask: 真实标签的二值mask，形状与pred_mask相同
        spacing: 图像的spacing
            - 2D: (spacing_y, spacing_x) 或 None
            - 3D: (spacing_x, spacing_y, spacing_z) 或 None
            单位为mm。如果为None，则使用像素单位。
    
    Returns:
        HD95值，单位为mm（如果提供了spacing）或像素（如果没有spacing）
    """
    # 两个都为空 -> 完美匹配
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0

    # 判断是2D还是3D
    is_3d = pred_mask.ndim == 3
    
    if is_3d:
        d, h, w = pred_mask.shape
        if spacing is not None:
            # 3D: spacing = (spacing_x, spacing_y, spacing_z)
            max_dist = float(np.sqrt((d * spacing[2]) ** 2 + (h * spacing[1]) ** 2 + (w * spacing[0]) ** 2))
        else:
            max_dist = float(np.sqrt(d ** 2 + h ** 2 + w ** 2))
    else:
        h, w = pred_mask.shape
        if spacing is not None:
            # 2D: spacing = (spacing_y, spacing_x)
            max_dist = float(np.sqrt((h * spacing[0]) ** 2 + (w * spacing[1]) ** 2))
        else:
            max_dist = float(np.sqrt(h ** 2 + w ** 2))

    # 一方为空 -> 返回上界
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return max_dist

    if not SCIPY_AVAILABLE:
        return float('nan')

    # 提取边界（foreground XOR eroded_foreground）
    pred_eroded = binary_erosion(pred_mask)
    gt_eroded = binary_erosion(gt_mask)
    pred_boundary = pred_mask ^ pred_eroded
    gt_boundary = gt_mask ^ gt_eroded

    pred_pts = np.column_stack(np.where(pred_boundary))
    gt_pts = np.column_stack(np.where(gt_boundary))

    if pred_pts.size == 0 or gt_pts.size == 0:
        return max_dist

    # 应用spacing（如果提供）将像素坐标转换为物理坐标
    if spacing is not None:
        pred_pts_physical = pred_pts.copy().astype(float)
        gt_pts_physical = gt_pts.copy().astype(float)
        
        if is_3d:
            # 3D: spacing = (spacing_x, spacing_y, spacing_z)
            # np.where返回的坐标顺序是 (z, y, x)
            pred_pts_physical[:, 0] *= spacing[2]  # z方向
            pred_pts_physical[:, 1] *= spacing[1]  # y方向
            pred_pts_physical[:, 2] *= spacing[0]  # x方向
            
            gt_pts_physical[:, 0] *= spacing[2]  # z方向
            gt_pts_physical[:, 1] *= spacing[1]  # y方向
            gt_pts_physical[:, 2] *= spacing[0]  # x方向
        else:
            # 2D: spacing = (spacing_y, spacing_x)
            # np.where返回的坐标顺序是 (y, x)
            pred_pts_physical[:, 0] *= spacing[0]  # y方向
            pred_pts_physical[:, 1] *= spacing[1]  # x方向
            
            gt_pts_physical[:, 0] *= spacing[0]  # y方向
            gt_pts_physical[:, 1] *= spacing[1]  # x方向
        
        # 使用物理坐标计算距离
        tree_gt = cKDTree(gt_pts_physical)
        d_pred_to_gt, _ = tree_gt.query(pred_pts_physical, k=1)
        p95_pred_to_gt = np.percentile(d_pred_to_gt, 95)

        tree_pred = cKDTree(pred_pts_physical)
        d_gt_to_pred, _ = tree_pred.query(gt_pts_physical, k=1)
        p95_gt_to_pred = np.percentile(d_gt_to_pred, 95)
    else:
        # 使用像素坐标计算距离
        tree_gt = cKDTree(gt_pts)
        d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
        p95_pred_to_gt = np.percentile(d_pred_to_gt, 95)

        tree_pred = cKDTree(pred_pts)
        d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
        p95_gt_to_pred = np.percentile(d_gt_to_pred, 95)

    return float(max(p95_pred_to_gt, p95_gt_to_pred))


def compute_nsd(pred_mask, gt_mask, spacing=None, threshold_mm=2.0):
    """计算Normalized Surface Distance (NSD)，使用指定的阈值（默认2mm），支持2D和3D。
    
    NSD定义为：在阈值范围内的表面点数量 / 总表面点数量
    
    Args:
        pred_mask: 预测的二值mask，可以是2D (H, W) 或 3D (D, H, W)
        gt_mask: 真实标签的二值mask，形状与pred_mask相同
        spacing: 图像的spacing
            - 2D: (spacing_y, spacing_x) 或 None
            - 3D: (spacing_x, spacing_y, spacing_z) 或 None
            单位为mm。如果为None，则使用像素单位。
        threshold_mm: 距离阈值，单位为mm（如果提供了spacing）或像素（如果没有spacing）
    
    Returns:
        NSD值，范围[0, 1]
    """
    # 两个都为空 -> 完美匹配
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0
    
    # 一方为空 -> NSD为0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0

    if not SCIPY_AVAILABLE:
        return float('nan')

    # 判断是2D还是3D
    is_3d = pred_mask.ndim == 3

    # 提取边界
    pred_eroded = binary_erosion(pred_mask)
    gt_eroded = binary_erosion(gt_mask)
    pred_boundary = pred_mask ^ pred_eroded
    gt_boundary = gt_mask ^ gt_eroded

    pred_pts = np.column_stack(np.where(pred_boundary))
    gt_pts = np.column_stack(np.where(gt_boundary))

    if pred_pts.size == 0 or gt_pts.size == 0:
        return 0.0

    # 应用spacing（如果提供）
    if spacing is not None:
        pred_pts_physical = pred_pts.copy().astype(float)
        gt_pts_physical = gt_pts.copy().astype(float)
        
        if is_3d:
            # 3D: spacing = (spacing_x, spacing_y, spacing_z)
            # np.where返回的坐标顺序是 (z, y, x)
            pred_pts_physical[:, 0] *= spacing[2]  # z方向
            pred_pts_physical[:, 1] *= spacing[1]  # y方向
            pred_pts_physical[:, 2] *= spacing[0]  # x方向
            
            gt_pts_physical[:, 0] *= spacing[2]  # z方向
            gt_pts_physical[:, 1] *= spacing[1]  # y方向
            gt_pts_physical[:, 2] *= spacing[0]  # x方向
        else:
            # 2D: spacing = (spacing_y, spacing_x)
            # np.where返回的坐标顺序是 (y, x)
            pred_pts_physical[:, 0] *= spacing[0]  # y方向
            pred_pts_physical[:, 1] *= spacing[1]  # x方向
            
            gt_pts_physical[:, 0] *= spacing[0]  # y方向
            gt_pts_physical[:, 1] *= spacing[1]  # x方向
        
        # 使用物理坐标
        tree_gt = cKDTree(gt_pts_physical)
        d_pred_to_gt, _ = tree_gt.query(pred_pts_physical, k=1)
        
        tree_pred = cKDTree(pred_pts_physical)
        d_gt_to_pred, _ = tree_pred.query(gt_pts_physical, k=1)
    else:
        # 使用像素坐标
        tree_gt = cKDTree(gt_pts)
        d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
        
        tree_pred = cKDTree(pred_pts)
        d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)

    # 计算双向NSD：分别计算两个方向的NSD，然后取平均
    # 对于pred的每个表面点，检查到gt表面的距离是否<=阈值
    pred_within_threshold = (d_pred_to_gt <= threshold_mm).sum()
    # 对于gt的每个表面点，检查到pred表面的距离是否<=阈值
    gt_within_threshold = (d_gt_to_pred <= threshold_mm).sum()
    
    # 计算两个方向的NSD
    if len(pred_pts) > 0:
        nsd_pred = pred_within_threshold / len(pred_pts)
    else:
        nsd_pred = 1.0 if len(gt_pts) == 0 else 0.0
    
    if len(gt_pts) > 0:
        nsd_gt = gt_within_threshold / len(gt_pts)
    else:
        nsd_gt = 1.0 if len(pred_pts) == 0 else 0.0
    
    # NSD = 两个方向的平均值（对称定义）
    nsd = (nsd_pred + nsd_gt) / 2.0
    return float(nsd)


def decode_rle_mask(rle, height, width):
    """解码COCO RLE格式的mask
    
    Args:
        rle: RLE编码，可以是字典格式或字符串格式
        height: mask的高度
        width: mask的宽度
    
    Returns:
        解码后的二值mask，形状为(height, width)
    """
    if isinstance(rle, dict):
        rle_obj = rle
    else:
        rle_obj = {
            "counts": rle,
            "size": [height, width]
        }
    mask = mask_utils.decode(rle_obj)
    return mask.astype(bool)


def load_spacing_map(spacing_file):
    """从JSON文件加载spacing映射
    
    JSON格式示例:
        {
            "patient0276_4CH_ED": [0.308, 0.308, 1.0],
            "patient0201_2CH_ED": [0.308, 0.308, 1.0]
        }
    
    Args:
        spacing_file: JSON文件路径，如果为None则返回空字典
    
    Returns:
        dict: 键为patient标识符，值为(spacing_x, spacing_y, spacing_z)元组
    """
    if spacing_file is None:
        return {}
    if not os.path.exists(spacing_file):
        print(f"Warning: spacing file not found: {spacing_file}")
        return {}

    try:
        with open(spacing_file, 'r') as f:
            data = json.load(f)
        spacing_map = {}
        for k, v in data.items():
            try:
                spacing_map[str(k)] = tuple(float(x) for x in v[:3])
            except Exception:
                continue
        return spacing_map
    except Exception as e:
        print(f"Warning: failed to load spacing JSON {spacing_file}: {e}")
        return {}
