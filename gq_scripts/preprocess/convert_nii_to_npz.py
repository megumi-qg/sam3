"""
convert_acdc_to_npz.py

将 ACDC/CAMUS 等数据集的 `.nii.gz` 格式图像和标注对转换为 BiomedParse 所需的 `.npz` 格式。

功能说明：
- 从包含 `img/` 和 `gt/` 子文件夹的目录读取图像和标注文件
- 对每个病例进行处理：
  * 使用 nibabel 加载 `.nii.gz` 文件
  * 自动检测深度轴（通常是最小的维度）并将其移动到第一维，得到 (D, H, W) 格式
  * 可选择移除所有标注为全零的背景切片（默认移除，可通过 --keep-all-slices 保留所有切片）
  * 将图像强度归一化并缩放到 [0, 255] 范围，转换为 uint8 类型
  * 保持标注为整数标签（int32 类型）
  * 提取并重新排序体素间距（spacing），使其与 (D, H, W) 轴顺序匹配
  * 保存到 `<output_root>/*.npz` 文件，包含以下键：
    - `imgs`: 处理后的图像数组，形状为 (D, H, W)，dtype=uint8
    - `gts`: 标注数组，形状为 (D, H, W)，dtype=int32
    - `spacing`: 体素间距元组，格式为 (depth, height, width)

使用示例：
python gq_scripts/preprocess/convert_nii_to_npz.py \
    --dataset-root /home/gaoqi/dataset/using/acdc2/test \
    --output-root /home/gaoqi/dataset/using/acdc4/test/data \
    --keep-all-slices

参数说明：
    --dataset-root: 数据集目录路径（必需），直接指向包含 img/ 和 gt/ 子文件夹的目录
    --output-root: 输出目录，默认为 dataset-root
    --keep-all-slices: 保留所有切片，不移除背景切片（默认会移除全零背景切片）
    --write-prompts: 可选标志，在数据目录中生成 class_prompts.json 模板文件

注意事项：
- 默认会移除所有标注为全零的背景切片；使用 --keep-all-slices 可保留所有切片
- 如果启用背景切片移除且移除后所有切片都被过滤掉，则跳过该病例
- 如果找不到对应的标注文件，会尝试多种命名模式（如 `*_gt.nii.gz`）进行匹配
- 处理过程中遇到错误会跳过该病例并继续处理下一个
- 输出文件使用压缩格式（npz_compressed）以节省存储空间
- 支持 ACDC 和 CAMUS 数据集的命名模式
"""
import os
import argparse
import glob
import numpy as np
import nibabel as nib


def detect_depth_axis(shape):
    # Heuristic: depth is typically the smallest axis for clinical MR/CT volumes
    # For 2D images (H, W), return None (no depth axis)
    if len(shape) == 2:
        return None
    return int(np.argmin(shape))


def reorder_spacing(zooms, depth_axis):
    # Return spacing ordered as (depth, height, width)
    # For 2D images, return (1.0, height, width)
    if depth_axis is None:
        # 2D image: add depth dimension with spacing 1.0
        if len(zooms) >= 2:
            return (1.0, float(zooms[0]), float(zooms[1]))
        else:
            return (1.0, 1.0, 1.0)
    
    axes = list(range(len(zooms)))
    new_axes = [depth_axis] + [ax for ax in axes if ax != depth_axis]
    return tuple(float(zooms[ax]) for ax in new_axes)


def process_case(img_path, gt_path, remove_background=True):
    """
    处理单个病例，将 .nii.gz 文件转换为处理后的数据字典。
    
    参数:
        img_path: 图像文件路径
        gt_path: 标注文件路径
        remove_background: 是否移除背景切片（全零标注的切片），默认为 True
    
    返回:
        包含 'imgs', 'gts', 'spacing' 的字典，如果移除背景后无有效切片则返回 None
    """
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)

    imgs = img_nii.get_fdata()
    gts = gt_nii.get_fdata()

    # Handle 2D images (some datasets like CAMUS have 2D ultrasound images)
    if imgs.ndim == 2:
        # Add depth dimension: (H, W) -> (1, H, W)
        imgs = imgs[np.newaxis, :, :]
        gts = gts[np.newaxis, :, :]
    elif imgs.ndim == 3:
        # 3D volume: determine depth axis and move it to axis 0 -> (D,H,W)
        depth_axis = detect_depth_axis(imgs.shape)
        if depth_axis is not None:
            imgs = np.moveaxis(imgs, depth_axis, 0)
            gts = np.moveaxis(gts, depth_axis, 0)
    else:
        raise ValueError(f"Expect 2D or 3D volumes. Got shapes imgs={imgs.shape}, gts={gts.shape}")

    # Ensure arrays are 3D now
    if imgs.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"Expect 3D volumes after processing. Got shapes imgs={imgs.shape}, gts={gts.shape}")

    # Convert types: gts -> int32, imgs -> float for scaling
    gts = gts.astype(np.int32)
    imgs = imgs.astype(np.float32)

    # Remove slices that are pure background (gts slice all zero) if requested
    if remove_background:
        mask_nonbg = np.any(gts != 0, axis=(1, 2))  # shape (D,)
        if not np.any(mask_nonbg):
            # nothing left after removal
            return None
        imgs = imgs[mask_nonbg]
        gts = gts[mask_nonbg]

    # Scale image intensities to [0,255] and cast to uint8
    mn = float(np.min(imgs))
    mx = float(np.max(imgs))
    if mx > mn:
        imgs = (imgs - mn) / (mx - mn)
    else:
        imgs = np.zeros_like(imgs)
    imgs = np.clip((imgs * 255.0), 0, 255).astype(np.uint8)

    # Reorder spacing to (D,H,W)
    zooms = img_nii.header.get_zooms()
    depth_axis = detect_depth_axis(imgs.shape) if imgs.shape[0] > 1 else None
    
    if depth_axis is None:
        # 2D image: spacing should be (1.0, height_spacing, width_spacing)
        if len(zooms) >= 2:
            spacing = (1.0, float(zooms[0]), float(zooms[1]))
        else:
            spacing = (1.0, 1.0, 1.0)
    else:
        # 3D image: need to reorder
        if len(zooms) >= 3:
            zooms3 = zooms[:3]
        else:
            zooms3 = zooms + (1.0,) * (3 - len(zooms))
        spacing = reorder_spacing(zooms3, depth_axis)

    return {"imgs": imgs, "gts": gts.astype(np.int32), "spacing": spacing}


def run(dataset_dir, output_dir, pattern="*.nii.gz", remove_background=True):
    """
    处理单个包含 img/ 和 gt/ 子文件夹的目录
    
    参数:
        dataset_dir: 包含 img/ 和 gt/ 子文件夹的目录路径
        output_dir: 输出目录路径
        pattern: 文件匹配模式，默认为 "*.nii.gz"
        remove_background: 是否移除背景切片
    """
    img_dir = os.path.join(dataset_dir, "img")
    gt_dir = os.path.join(dataset_dir, "gt")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, pattern)))
    if len(img_files) == 0:
        print(f"No image files found in {img_dir} matching {pattern}")
        return 0, 0

    n_saved = 0
    n_skipped = 0

    for img_path in img_files:
        base = os.path.basename(img_path)
        name = os.path.splitext(os.path.splitext(base)[0])[0]
        # Try to find matching GT file. common pattern: same basename or basename + '_gt'
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            base_no_ext = name
            alt1 = os.path.join(gt_dir, f"{base_no_ext}_gt.nii.gz")
            alt2 = os.path.join(gt_dir, f"{base_no_ext}_gt.nii")
            found = None
            if os.path.exists(alt1):
                found = alt1
            elif os.path.exists(alt2):
                found = alt2
            else:
                # search for any file in gt_dir containing the base and '_gt'
                for p in sorted(glob.glob(os.path.join(gt_dir, "*"))):
                    fname = os.path.basename(p)
                    if base_no_ext in fname and "_gt" in fname and fname.endswith((".nii.gz", ".nii")):
                        found = p
                        break

            if found is None:
                print(f"GT not found for {base}, skipping")
                n_skipped += 1
                continue
            else:
                gt_path = found

        try:
            processed = process_case(img_path, gt_path, remove_background=remove_background)
        except Exception as e:
            print(f"Error processing {base}: {e}")
            n_skipped += 1
            continue

        if processed is None:
            if remove_background:
                print(f"All slices are background after filtering for {base}, skipping")
            else:
                print(f"Failed to process {base}, skipping")
            n_skipped += 1
            continue

        out_path = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(out_path, imgs=processed["imgs"], gts=processed["gts"], spacing=processed["spacing"])
        print(f"Saved {out_path}: imgs shape {processed['imgs'].shape}, gts shape {processed['gts'].shape}")
        n_saved += 1

    print(f"Done. Saved {n_saved} files, skipped {n_skipped} files.")
    return n_saved, n_skipped


def make_prompts_template(output_data_root, dataset_name="ACDC"):
    # create a minimal class_prompts.json template for ACDC
    import json
    prompts = {
        dataset_name: {
            "1": ["Right ventricle", "Right ventricle in cardiac MR"],
            "2": ["Myocardium", "Myocardium in cardiac MR"],
            "3": ["Left ventricle", "Left ventricle in cardiac MR"],
            "instance_label": 0
        }
    }
    out_path = os.path.join(output_data_root, "class_prompts.json")
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote class_prompts.json to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ACDC/CAMUS .nii.gz files to BiomedParse .npz format")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to dataset directory containing img/ and gt/ subdirectories")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Output directory; defaults to dataset-root")
    parser.add_argument("--keep-all-slices", action="store_true",
                        help="Keep all slices including background slices (default: remove background slices)")
    parser.add_argument("--write-prompts", action="store_true", 
                        help="Write a minimal class_prompts.json template to the parent data folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_root = args.output_root if args.output_root is not None else args.dataset_root
    # remove_background is True by default, set to False if --keep-all-slices is specified
    remove_background = not args.keep_all_slices
    
    # 处理包含 img/ 和 gt/ 的目录
    run(args.dataset_root, out_root, remove_background=remove_background)
    
    if args.write_prompts:
        # place prompts in parent data folder (one level up from dataset_root)
        data_root = os.path.dirname(args.dataset_root)
        make_prompts_template(data_root, os.path.basename(args.dataset_root))
