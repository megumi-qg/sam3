import torch
import os

# 加载微调后的检查点文件
# 可以选择原始检查点或转换后的检查点
# CHECKPOINT_PATH = '/home/gaoqi/sam3/experiment/acdc_weak_finetune_3/checkpoints/checkpoint.pt'
CHECKPOINT_PATH = '/home/gaoqi/sam3/gq_experiment/acdc/lora/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt'

print(f"正在加载: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

# 检查是否存在 'model' 键
if 'model' not in ckpt:
    print("错误: 检查点中未找到 'model' 键")
    exit(1)

# 提取 model 键下的 state_dict
model_state_dict = ckpt['model']

# 构造输出文件名：在原文件名基础上添加 '_model_only.pt'
base_name = os.path.basename(CHECKPOINT_PATH)
name_without_ext = os.path.splitext(base_name)[0]
output_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), f"{name_without_ext}_model_only.pt")

print(f"正在保存 model 到: {output_path}")

# 保存 model state_dict
torch.save(model_state_dict, output_path)

print("保存成功！")
print(f"Model 参数数量: {len(model_state_dict)}")