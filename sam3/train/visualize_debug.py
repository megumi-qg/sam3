import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os

# Set SAM3_VIS_IMAGE, SAM3_VIS_CKPT (required), SAM3_VIS_CONFIG optional.
IMAGE_PATH = os.environ.get("SAM3_VIS_IMAGE", "")
CHECKPOINT_PATH = os.environ.get("SAM3_VIS_CKPT", "")
CONFIG_NAME = os.environ.get("SAM3_VIS_CONFIG", "configs/acdc/train.yaml")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def prepare_image(image_path, resolution=1024):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]

    # 简单的预处理 (Resize + Normalize)
    # 注意：这里模拟 Resize 到 1024 的操作，实际 transform 更复杂，但可视化足够了
    scale = resolution / max(original_size)
    new_w = int(original_size[1] * scale)
    new_h = int(original_size[0] * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 归一化 (匹配配置中的 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    
    # Pad 到 1024x1024 (这也是模型预期的)
    c, h, w = img_tensor.shape
    pad_h = resolution - h
    pad_w = resolution - w
    img_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), value=0)

    return img, img_padded.unsqueeze(0), scale

def main():
    if not IMAGE_PATH or not CHECKPOINT_PATH:
        raise SystemExit(
            "Set SAM3_VIS_IMAGE and SAM3_VIS_CKPT to an image file and checkpoint path."
        )
    if GlobalHydra().is_initialized():
        GlobalHydra().instance().clear()
    
    initialize(config_path=".", version_base=None)
    cfg = compose(config_name=CONFIG_NAME)
    
    # 2. 构建模型
    print(">>>正在加载模型...")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH,
        enable_segmentation=True,
        eval_mode=True,
        device='cuda'
    )
    model = model.to('cuda')
    model.eval()
    print(">>> 模型加载完成")

    # 3. 准备数据
    original_img, input_tensor, scale = prepare_image(IMAGE_PATH)
    input_tensor = input_tensor.to('cuda')

    # 4. 推理
    print(">>> 正在推理...")
    with torch.no_grad():
        # SAM3 的前向传播，这里不给 prompt，测试它的自动检测能力
        outputs = model(input_tensor)

    # 5. 分析输出 (Debug 核心)
    # outputs 通常包含 'pred_logits', 'pred_boxes', 'pred_masks'
    pred_logits = outputs['pred_logits'][0] # (MQ, NumClasses)
    pred_boxes = outputs['pred_boxes'][0]   # (MQ, 4)
    pred_masks = outputs['pred_masks'][0]   # (MQ, H, W)

    # 获取概率最高的 Top K
    scores = pred_logits.sigmoid().max(dim=-1)[0]
    # 设定一个置信度阈值，看看有多少通过了
    threshold = 0.3 
    keep = scores > threshold
    
    print("\n" + "="*30)
    print(f"DEBUG 信息:")
    print(f"原始查询数量: {len(scores)}")
    print(f"置信度 > {threshold} 的数量: {keep.sum().item()}")
    
    if keep.sum() == 0:
        print("警告: 没有检测到任何目标！可能是阈值太高，或者模型没收敛。")
        print(f"最高置信度为: {scores.max().item():.4f}")
        top_idx = scores.argmax()
    else:
        top_idx = torch.where(keep)[0]

    # 打印前几个预测的类别
    probs = pred_logits.sigmoid()
    top_scores, top_classes = probs.max(dim=-1)
    
    print("\nTop 5 预测 (即使 score 很低):")
    values, indices = torch.topk(top_scores, 5)
    for i in indices:
        print(f"Index {i}: Class ID={top_classes[i].item()}, Score={top_scores[i].item():.4f}")
    print("="*30 + "\n")

    # 6. 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img)
    
    # 只需要画置信度高的
    if keep.sum() > 0:
        for i in torch.where(keep)[0]:
            # 还原 Box 坐标 (cx, cy, w, h) -> (x0, y0, x1, y1)
            # 注意：SAM3 输出的是归一化的 (cx, cy, w, h)
            box = pred_boxes[i].cpu().numpy()
            H, W = input_tensor.shape[-2:]
            cx, cy, w, h = box * np.array([W, H, W, H])
            x0 = (cx - 0.5 * w) / scale
            y0 = (cy - 0.5 * h) / scale
            w = w / scale
            h = h / scale
            
            mask = pred_masks[i].cpu().numpy()
            # Mask 通常是 256x256 或低分辨率的，需要插值回原图
            # 这里简单处理，只可视化 box
            show_box([x0, y0, x0+w, y0+h], plt.gca())
            
            # Mask 处理 (简单阈值)
            mask_prob = torch.sigmoid(pred_masks[i])
            mask_binary = (mask_prob > 0.5).cpu().numpy()
            # 这里的 mask 也是 pad 过后的尺寸，需要 crop 和 resize 回去 (略复杂，暂时看 Box 和 Logits 就够了)
            # show_mask(mask_binary, plt.gca())
            
            plt.text(x0, y0, f"Cls:{top_classes[i].item()} {top_scores[i].item():.2f}", 
                     color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    save_path = "debug_vis.png"
    plt.savefig(save_path)
    print(f"可视化结果已保存至: {save_path}")

if __name__ == "__main__":
    main()