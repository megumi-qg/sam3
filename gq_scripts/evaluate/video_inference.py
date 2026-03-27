import os
import sys
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import argparse
import tempfile
import shutil
import random
from collections import OrderedDict
from PIL import Image

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed_for_reproducibility(seed=42):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置CUDA确定性选项
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed} (确定性模式)")

def visualize_frame_with_scores(
    frame_idx,
    video_frames,
    outputs_dict,
    outputs_with_scores_dict,
    title="SAM 3 Dense Tracking outputs",
    figsize=(12, 8),
):
    """
    可视化视频帧的分割结果，并显示每个对象的score
    
    Args:
        frame_idx: 要可视化的帧索引
        video_frames: 视频帧列表（numpy数组或文件路径列表）
        outputs_dict: {frame_idx: {obj_id: mask}} 格式的输出字典
        outputs_with_scores_dict: {frame_idx: {'out_obj_ids': tensor, 'out_probs': tensor, ...}} 格式的原始输出
        title: 图表标题
        figsize: 图表大小
    """
    from sam3.visualization_utils import load_frame, COLORS, plot_bbox, plot_mask, masks_to_boxes, normalize_bbox
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(f"Frame {frame_idx} - {title}")
    
    # 加载图像
    img = load_frame(video_frames[frame_idx])
    img_H, img_W, _ = img.shape
    ax.imshow(img)
    
    # 获取该帧的输出
    if frame_idx in outputs_dict:
        _outputs = outputs_dict[frame_idx]
    else:
        ax.text(
            0.5, 0.5, "No objects detected",
            transform=ax.transAxes, fontsize=16, ha="center", va="center",
            color="red", weight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        return
    
    # 获取score信息
    score_dict = {}
    if frame_idx in outputs_with_scores_dict:
        raw_output = outputs_with_scores_dict[frame_idx]
        if "out_obj_ids" in raw_output and "out_probs" in raw_output:
            # 处理obj_ids
            obj_ids_tensor = raw_output["out_obj_ids"]
            if isinstance(obj_ids_tensor, torch.Tensor):
                obj_ids = obj_ids_tensor.cpu().tolist()
            elif isinstance(obj_ids_tensor, np.ndarray):
                obj_ids = obj_ids_tensor.tolist()
            else:
                obj_ids = list(obj_ids_tensor)
            
            # 处理probs
            probs_tensor = raw_output["out_probs"]
            if isinstance(probs_tensor, torch.Tensor):
                probs = probs_tensor.cpu().tolist()
            elif isinstance(probs_tensor, np.ndarray):
                probs = probs_tensor.tolist()
            else:
                probs = list(probs_tensor)
            
            # 构建score字典
            score_dict = {int(obj_id): float(prob) for obj_id, prob in zip(obj_ids, probs)}
    
    # 绘制每个对象
    objects_drawn = 0
    for obj_id, binary_mask in _outputs.items():
        mask_sum = (
            binary_mask.sum()
            if hasattr(binary_mask, "sum")
            else np.sum(binary_mask)
        )
        
        if mask_sum > 0:  # Only draw if mask has content
            # Convert to torch tensor if it's not already
            if not isinstance(binary_mask, torch.Tensor):
                binary_mask = torch.tensor(binary_mask)
            
            # Find bounding box from mask
            if binary_mask.any():
                box_xyxy = masks_to_boxes(binary_mask.unsqueeze(0)).squeeze()
                box_xyxy = normalize_bbox(box_xyxy, img_W, img_H)
            else:
                box_xyxy = [0.45, 0.45, 0.55, 0.55]
            
            color = COLORS[obj_id % len(COLORS)]
            
            # 构建显示文本：包含obj_id和score（如果有）
            if obj_id in score_dict:
                score_text = f"(id={obj_id}, score={score_dict[obj_id]:.2f})"
            else:
                score_text = f"(id={obj_id})"
            
            plot_bbox(
                img_H,
                img_W,
                box_xyxy,
                text=score_text,
                box_format="XYXY",
                color=color,
                ax=ax,
            )
            
            # Convert back to numpy for plotting
            mask_np = (
                binary_mask.numpy()
                if isinstance(binary_mask, torch.Tensor)
                else binary_mask
            )
            plot_mask(mask_np, color=color, ax=ax)
            objects_drawn += 1
    
    if objects_drawn == 0:
        ax.text(
            0.5, 0.5, "No objects detected",
            transform=ax.transAxes, fontsize=16, ha="center", va="center",
            color="red", weight="bold"
        )
    
    ax.axis("off")
    plt.tight_layout()

"""
使用示例：

1. JPEG文件夹或MP4文件：
python gq_scripts/evaluate/video_inference.py \
    --checkpoint /home/gaoqi/sam3/gq_experiment/acdc_camus/weak/scribble/checkpoints/val_acdc_segm_model_merge.pt \
    --video /home/gaoqi/sam3/gq_scripts/evaluate/example/acdc_p01_f01 \
    --prompt "right ventricle" \
    --frame_idx 0 \
    --output_dir /home/gaoqi/sam3/gq_scripts/outputs \
    --vis_stride 1

2. NPZ格式3D数据：
python gq_scripts/evaluate/video_inference.py \
    --checkpoint /home/gaoqi/sam3/gq_experiment/acdc_camus/full/1/checkpoints/val_acdc_segmentation_coco_eval_segm_AP_model_merge.pt \
    --video /home/gaoqi/dataset/using/acdc4/test/data/patient101_frame01.npz \
    --prompt "Right Ventricle" \
    --frame_idx 5 \
    --output_dir /home/gaoqi/sam3/gq_scripts/outputs \
    --vis_stride 1 \
    --keep_temp_frames  # 可选：保留临时帧文件
"""
#########################
##   MAIN  sam3 视频推理脚本
#########################

def main():
    parser = argparse.ArgumentParser(description='SAM3视频推理脚本')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/acdc_camus/weak/scribble/checkpoints/merged_model_with_tracker.pt",
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='视频路径（JPEG文件夹、MP4文件或NPZ文件）'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="right ventricle",
        help='文本提示（例如："right ventricle"）'
    )
    parser.add_argument(
        '--frame_idx',
        type=int,
        default=0,
        help='添加提示的帧索引（默认：0）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认为脚本目录下的outputs文件夹）'
    )
    parser.add_argument(
        '--vis_stride',
        type=int,
        default=None,
        help='可视化帧间隔（默认：总帧数/10）'
    )
    parser.add_argument(
        '--keep_temp_frames',
        action='store_true',
        help='保留从npz文件生成的临时帧文件（默认会删除）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='随机种子（默认：42），用于确保结果可复现'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='启用确定性模式（默认：True），确保结果可复现'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可复现
    if args.deterministic:
        set_seed_for_reproducibility(args.seed)
    else:
        print("警告: 未启用确定性模式，结果可能不可复现")
    
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    print("sam3_root:", sam3_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # - Build model
    print("Loading model ...")
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = args.checkpoint

    # 预处理检查点：处理 model 键和 detector/tracker 前缀
    print(f"预处理检查点文件: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 提取 model 键（如果存在）
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # 检查键是否包含 "detector" 和 "tracker" 前缀
    keys = list(state_dict.keys())
    has_detector_prefix = any("detector" in k for k in keys)
    has_tracker_prefix = any("tracker" in k for k in keys)

    print(f"检查点键前缀情况: detector={has_detector_prefix}, tracker={has_tracker_prefix}")

    # 如果checkpoint没有detector/tracker前缀，需要手动处理
    # 但build_sam3_video_model期望有这些前缀，所以我们需要确保checkpoint格式正确
    # 如果checkpoint已经有detector和tracker前缀，直接使用
    # 如果没有，需要创建一个临时checkpoint文件

    if not has_detector_prefix or not has_tracker_prefix:
        print("警告: 检查点格式可能不符合视频模型要求")
        print("视频模型期望checkpoint包含 'detector.' 和 'tracker.' 前缀")
        if not has_detector_prefix:
            print("检测到键没有 'detector.' 前缀，自动添加...")
            processed_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if not k.startswith("tracker."):
                    new_key = "detector." + k
                    processed_state_dict[new_key] = v
                else:
                    processed_state_dict[k] = v
            state_dict = processed_state_dict
            print("已添加 'detector.' 前缀")
    else:
        print("检查点已包含 'detector.' 和 'tracker.' 前缀，格式正确")

    # 构建视频预测器
    # 注意：build_sam3_video_predictor会调用build_sam3_video_model
    # build_sam3_video_model期望checkpoint有detector和tracker前缀
    print("构建视频预测器...")
    gpus_to_use = [torch.cuda.current_device()] if torch.cuda.is_available() else None

    # 如果checkpoint格式正确，直接使用
    # 否则需要先保存处理后的checkpoint到临时文件
    if has_detector_prefix and has_tracker_prefix:
        # checkpoint格式正确，直接使用
        final_checkpoint_path = checkpoint_path
    else:
        # 创建临时checkpoint文件
        # 注意：如果原始checkpoint有"model"键，我们需要保持相同的结构
        temp_ckpt_path = checkpoint_path.replace(".pt", "_temp_video.pt")
        print(f"创建临时checkpoint文件: {temp_ckpt_path}")
        # 保存处理后的state_dict（build_sam3_video_model会处理"model"键的情况）
        torch.save(state_dict, temp_ckpt_path)
        final_checkpoint_path = temp_ckpt_path
        print("注意: 临时文件将在脚本结束后保留，可手动删除")

    # 构建视频预测器
    predictor = build_sam3_video_predictor(
        checkpoint_path=final_checkpoint_path,
        bpe_path=bpe_path,
        gpus_to_use=gpus_to_use,
        strict_state_dict_loading=False,
    )

    print("模型加载完成！")
    
    # 确保模型处于eval模式（虽然build_sam3_video_predictor应该已经设置了）
    # 但为了确保确定性，我们再次确认
    if hasattr(predictor, 'model'):
        predictor.model.eval()
        print("已确认模型处于eval模式")

    # - Load video
    print("Loading video ...")
    # video_path可以是JPEG文件夹、MP4视频文件或NPZ文件
    video_path = args.video
    temp_npz_dir = None  # 用于存储从npz生成的临时帧文件

    # 检查是否为npz文件
    if isinstance(video_path, str) and video_path.endswith(".npz"):
        print(f"检测到NPZ文件: {video_path}")
        print("正在加载NPZ数据...")
        
        # 加载npz文件
        npz_data = np.load(video_path)
        
        # 检查必需的键
        required_keys = ['imgs', 'gts', 'spacing']
        missing_keys = [k for k in required_keys if k not in npz_data]
        if missing_keys:
            raise ValueError(f"NPZ文件缺少必需的键: {missing_keys}")
        
        imgs = npz_data['imgs']  # (D, H, W), uint8
        gts = npz_data['gts']    # (D, H, W), int32
        spacing = npz_data['spacing']  # (depth, height, width)
        
        print(f"NPZ数据形状: imgs={imgs.shape}, gts={gts.shape}, spacing={spacing}")
        
        # 创建临时目录存储帧文件
        temp_npz_dir = tempfile.mkdtemp(prefix="sam3_npz_frames_")
        print(f"创建临时目录存储帧文件: {temp_npz_dir}")
        
        # 将每个depth切片转换为RGB图像并保存
        num_slices = imgs.shape[0]
        print(f"正在将 {num_slices} 个切片转换为图像文件...")
        
        for slice_idx in range(num_slices):
            # 获取单个切片 (H, W)
            slice_img = imgs[slice_idx]
            
            # 确保是uint8类型
            if slice_img.dtype != np.uint8:
                slice_img = slice_img.astype(np.uint8)
            
            # 转换为PIL Image
            img_pil = Image.fromarray(slice_img, mode='L')  # 灰度图像
            
            # 转换为RGB（通过复制灰度通道到3个通道）
            img_rgb = img_pil.convert('RGB')
            
            # 保存为PNG文件，文件名格式为 <slice_idx>.png
            frame_filename = f"{slice_idx:05d}.png"
            frame_path = os.path.join(temp_npz_dir, frame_filename)
            img_rgb.save(frame_path)
        
        print(f"已将 {num_slices} 个切片保存到临时目录")
        
        # 更新video_path为临时目录（SAM3预测器需要文件路径）
        video_path = temp_npz_dir
        
        # 预加载所有帧用于可视化
        print("预加载帧用于可视化...")
        video_frames_for_vis = []
        for slice_idx in range(num_slices):
            frame_filename = f"{slice_idx:05d}.png"
            frame_path = os.path.join(temp_npz_dir, frame_filename)
            img = Image.open(frame_path)
            video_frames_for_vis.append(np.array(img))
        
        print(f"已预加载 {len(video_frames_for_vis)} 帧")
    
    # 加载视频帧用于可视化
    elif isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        # 假设是JPEG文件夹
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        if not video_frames_for_vis:
            video_frames_for_vis = glob.glob(os.path.join(video_path, "*.png"))
        try:
            # 整数排序而不是字符串排序（例如 "2.jpg" 在 "11.jpg" 之前）
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # 如果格式不是 "<frame_index>.jpg"，回退到字典序排序
            print(
                f'帧名称不是 "<frame_index>.jpg" 格式: {video_frames_for_vis[:5]}, '
                f"回退到字典序排序。"
            )
            video_frames_for_vis.sort()
        
        # 预加载图像并确保是RGB格式（3通道）
        # 这样可以避免在可视化时出现维度错误
        print("预加载图像并转换为RGB格式...")
        loaded_frames = []
        for frame_path in video_frames_for_vis:
            img = Image.open(frame_path)
            # 如果是灰度图像，转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            loaded_frames.append(np.array(img))
        video_frames_for_vis = loaded_frames

    print(f"找到 {len(video_frames_for_vis)} 帧")

    # - Start session
    print("启动推理会话...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"会话ID: {session_id}")

    # - Reset session (如果需要)
    # 如果之前已经运行过文本提示，需要重置会话
    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    # - Add text prompt
    print("添加文本提示...")
    prompt_text_str = args.prompt
    frame_idx = args.frame_idx

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    out = response["outputs"]

    # 可视化第一帧的结果
    plt.close("all")
    # 准备带score的可视化数据
    first_frame_outputs = prepare_masks_for_visualization({frame_idx: out})
    first_frame_outputs_with_scores = {frame_idx: out}  # 保留原始输出以获取score
    
    # 创建自定义可视化函数来显示score
    visualize_frame_with_scores(
        frame_idx,
        video_frames_for_vis,
        first_frame_outputs,
        first_frame_outputs_with_scores,
        title="SAM 3 Dense Tracking outputs",
        figsize=(12, 8),
    )

    # 保存第一帧的结果
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"video_inference_frame_{frame_idx}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存第 {frame_idx} 帧结果到: {output_path}")

    # - Propagate in video
    print("传播到整个视频...")
    def propagate_in_video(predictor, session_id):
        """将输出从当前帧传播到视频末尾"""
        outputs_per_frame = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
        return outputs_per_frame

    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"已处理 {len(outputs_per_frame)} 帧")

    # - Visualize results
    print("可视化结果...")
    # 保存原始的outputs（包含score信息）用于可视化
    outputs_per_frame_with_scores = outputs_per_frame.copy()
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # 每隔一定帧数可视化一次
    if args.vis_stride is None:
        vis_frame_stride = max(1, len(outputs_per_frame) // 10)  # 最多显示10帧
    else:
        vis_frame_stride = args.vis_stride
    print(f"可视化间隔: 每 {vis_frame_stride} 帧")

    for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
        plt.close("all")
        # 获取该帧的原始输出（包含score）
        frame_outputs = outputs_per_frame[frame_idx] if frame_idx in outputs_per_frame else {}
        frame_outputs_with_scores = outputs_per_frame_with_scores.get(frame_idx, {})
        
        # 创建该帧的输出字典
        frame_outputs_dict = {frame_idx: frame_outputs}
        frame_outputs_with_scores_dict = {frame_idx: frame_outputs_with_scores}
        
        visualize_frame_with_scores(
            frame_idx,
            video_frames_for_vis,
            frame_outputs_dict,
            frame_outputs_with_scores_dict,
            title="SAM 3 Dense Tracking outputs",
            figsize=(12, 8),
        )
        
        # 保存结果
        output_path = os.path.join(output_dir, f"video_inference_frame_{frame_idx}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"已保存第 {frame_idx} 帧结果到: {output_path}")

    # - Close session
    print("关闭会话...")
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    # - Shutdown predictor
    print("关闭预测器...")
    predictor.shutdown()

    print("视频推理完成！")

if __name__ == "__main__":
    main()
