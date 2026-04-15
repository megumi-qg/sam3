"""
工具函数模块 - 用于batch_evaluate.py

包含评估指标计算、数据加载等工具函数：
- IoU、Dice、HD95、NSD等评估指标计算（HD95/NSD 使用 surface_distance 库）
- COCO格式mask解码
- Spacing映射文件加载
"""

import json
import os
import numpy as np
from pycocotools import mask as mask_utils

# surface_distance 用于 HD95/NSD（与 nnU-Net segmentation_metrics 一致）
try:
    from surface_distance import metrics as surf_metrics
    SURFACE_DISTANCE_AVAILABLE = True
except ImportError:
    surf_metrics = None
    SURFACE_DISTANCE_AVAILABLE = False


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
    """计算 95th percentile Hausdorff Distance (HD95)，使用 surface_distance 库（与 nnU-Net segmentation_metrics 一致）。
    
    仅支持 3D mask。spacing 必须与数组维度顺序一致：数组形状为 (D, H, W) 时，
    spacing 应为 (spacing_D, spacing_H, spacing_W)，单位为 mm。
    
    Args:
        pred_mask: 预测的二值 mask，3D (D, H, W)
        gt_mask: 真实标签的二值 mask，形状与 pred_mask 相同
        spacing: 体素间距，与数组轴顺序一致 (spacing_axis0, spacing_axis1, spacing_axis2)，单位 mm。
                 例如数组 (num_slices, h, w) 即 (z,y,x) 时，应传入 (spacing_z, spacing_y, spacing_x)。
    
    Returns:
        HD95 值 (mm)。空 mask 时返回 0.0（两者都空）或 np.inf（一方为空）。
    """
    if not SURFACE_DISTANCE_AVAILABLE:
        raise ImportError("需要安装 surface-distance 库: pip install surface-distance")
    
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    
    if pred_mask.ndim != 3:
        raise ValueError("compute_hd95 仅支持 3D mask，当前 ndim=%s" % pred_mask.ndim)
    
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return np.inf
    
    if spacing is None:
        spacing = (1.0,) * 3
    spacing_list = list(spacing)
    surface_distances = surf_metrics.compute_surface_distances(
        pred_mask, gt_mask, spacing_list
    )
    return float(surf_metrics.compute_robust_hausdorff(surface_distances, 95))


def compute_nsd(pred_mask, gt_mask, spacing=None, threshold_mm=2.0):
    """计算 Normalized Surface Dice (NSD)，使用 surface_distance 库（与 nnU-Net segmentation_metrics 一致）。
    
    仅支持 3D mask。spacing 必须与数组维度顺序一致：数组形状为 (D, H, W) 时，
    spacing 应为 (spacing_D, spacing_H, spacing_W)，单位为 mm。
    
    Args:
        pred_mask: 预测的二值 mask，3D (D, H, W)
        gt_mask: 真实标签的二值 mask，形状与 pred_mask 相同
        spacing: 体素间距，与数组轴顺序一致，单位 mm
        threshold_mm: 容差距离 (mm)，默认 2.0
    
    Returns:
        NSD 值 [0, 1]。两者都为空时返回 1.0，一方为空时返回 0.0。
    """
    if not SURFACE_DISTANCE_AVAILABLE:
        raise ImportError("需要安装 surface-distance 库: pip install surface-distance")
    
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    
    if pred_mask.ndim != 3:
        raise ValueError("compute_nsd 仅支持 3D mask，当前 ndim=%s" % pred_mask.ndim)
    
    if not pred_mask.any() and not gt_mask.any():
        return 1.0
    if not pred_mask.any() or not gt_mask.any():
        return 0.0
    
    if spacing is None:
        spacing = (1.0,) * 3
    spacing_list = list(spacing)
    surface_distances = surf_metrics.compute_surface_distances(
        pred_mask, gt_mask, spacing_list
    )
    return float(
        surf_metrics.compute_surface_dice_at_tolerance(
            surface_distances, threshold_mm
        )
    )


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
    
    JSON 通常由 nii 的 header.get_zooms() 生成，顺序为 nii 数组轴 (axis0, axis1, axis2)。
    常见 nii 形状为 (H, W, Z)，故 JSON 为 [spacing_H, spacing_W, spacing_Z]。
    
    JSON格式示例:
        {"patient101_frame01": [1.64, 1.64, 10.0], "Case00": [0.625, 0.625, 3.6]}
    
    Args:
        spacing_file: JSON文件路径，如果为None则返回空字典
    
    Returns:
        dict: 键为 patient 标识符，值为 (axis0, axis1, axis2) 的 3 元组（与 nii zooms 顺序一致）
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


def _nii_basename_to_key(basename, dataset_name):
    """从 .nii.gz 文件名得到与 batch_evaluate 中 spacing 查找一致的 key。
    
    dataset_name: 'acdc' | 'btcv' | 'promise12' | 'mms2' | 'camus' 等
    """
    name = basename[:-7] if basename.lower().endswith('.nii.gz') else basename
    d = (dataset_name or '').strip().lower()
    if d == 'btcv':
        # 0507688-Image.nii.gz -> 0507688
        if '-Image' in name:
            return name.split('-Image')[0]
        return name
    if d == 'promise12':
        # Case00.nii.gz -> Case00
        return name
    # ACDC: patient001_frame01; MMs2: 501_SA_ED; CAMUS: patient0001_2CH_ED 等，直接用文件名
    return name


def build_spacing_map_from_nii_dir(test_dir, dataset_name):
    """从 test_dir 及其子目录中的 .nii.gz 文件读取 spacing，构建 key -> (spacing_x, spacing_y, spacing_z)。
    
    会扫描 test_dir 以及 test_dir 下常见子目录（如 img, images, nii 等）中的 .nii.gz。
    键的规则与 batch_evaluate 中一致，便于用 (patient_id, frame_id) 推导的 key 查找。
    
    Args:
        test_dir: 测试集根目录
        dataset_name: 数据集名称，用于解析文件名得到 key（btcv / promise12 / acdc / mms2 / camus）
    
    Returns:
        dict: key -> (spacing_x, spacing_y, spacing_z)，与 load_spacing_map 返回格式一致。
              若未安装 nibabel 或未找到任何 nii，返回空字典。
    """
    try:
        import nibabel as nib
    except ImportError:
        return {}

    collected = {}
    # 扫描根目录及一层子目录
    to_scan = [test_dir]
    try:
        for entry in os.listdir(test_dir):
            path = os.path.join(test_dir, entry)
            if os.path.isdir(path) and entry.lower() in ('img', 'images', 'nii', 'volumes'):
                to_scan.append(path)
    except OSError:
        pass

    for dirpath in to_scan:
        if not os.path.isdir(dirpath):
            continue
        try:
            names = os.listdir(dirpath)
        except OSError:
            continue
        for fn in names:
            if not fn.lower().endswith('.nii.gz'):
                continue
            path = os.path.join(dirpath, fn)
            if not os.path.isfile(path):
                continue
            try:
                img = nib.load(path)
                zooms = img.header.get_zooms()
                if len(zooms) < 3:
                    zooms = (float(zooms[0]) if len(zooms) > 0 else 1.0,
                             float(zooms[1]) if len(zooms) > 1 else 1.0,
                             1.0)
                else:
                    zooms = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
            except Exception:
                continue
            key = _nii_basename_to_key(fn, dataset_name)
            if key:
                collected[key] = zooms  # (x, y, z) 与 NIfTI 前三维一致，和 load_spacing_map 格式一致

    return collected