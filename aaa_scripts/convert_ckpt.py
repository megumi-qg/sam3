import torch
from collections import OrderedDict
import os

# ================= 配置 =================
# 你的原始微调 checkpoint 路径
CHECKPOINT_PATH = "/home/gaoqi/sam3/experiment/acdc_finetune/checkpoints/checkpoint.pt"
# 输出路径
SAVE_PATH = CHECKPOINT_PATH.replace(".pt", "_converted.pt")
# =======================================

def convert():
    print(f"正在加载: {CHECKPOINT_PATH}")
    # 尝试加载，兼容不同的保存方式
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # 获取 state_dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 检查是否需要转换
    keys = list(state_dict.keys())
    if any(k.startswith("detector.") for k in keys):
        print("警告: 看起来这个 checkpoint 已经包含 'detector.' 前缀了。")
        print("前5个 keys:", keys[:5])
        user_input = input("是否强制继续添加前缀? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    print(f"检测到 {len(keys)} 个参数键。正在添加 'detector.' 前缀...")
    
    # 执行转换
    new_state_dict = OrderedDict()
    # 这里的关键是：SAM3 推理时预期整个模型都在 "detector" 命名空间下
    for k, v in state_dict.items():
        # 防止重复添加
        if not k.startswith("detector."):
            new_key = "detector." + k
        else:
            new_key = k
        new_state_dict[new_key] = v

    # 重新包装 (为了保持格式一致，通常包在 'model' 键里)
    save_dict = {"model": new_state_dict}
    
    # 复制其他元数据 (epoch, args 等) 如果有的话
    for k, v in checkpoint.items():
        if k not in ["model", "state_dict"]:
            save_dict[k] = v

    torch.save(save_dict, SAVE_PATH)
    print(f"转换成功！新权重已保存至: {SAVE_PATH}")
    print("现在请在 acdc_eval.yaml 或 visualize_debug.py 中使用这个新的路径。")

if __name__ == "__main__":
    convert()