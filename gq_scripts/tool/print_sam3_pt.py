import torch

# 加载微调后的检查点文件
# 可以选择原始检查点或转换后的检查点
# /home/gaoqi/official_ckpt/sam3_hf/sam3.pt
# /home/gaoqi/sam3/gq_experiment/acdc_camus/weak/scribble/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt
CHECKPOINT_PATH = '/home/gaoqi/sam3/gq_experiment/acdc_camus/weak/scribble/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt'

print(f"正在加载: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

# 显示检查点的顶层键
print("\n=== 检查点顶层键 ===")
for key in ckpt.keys():
    print(f"- {key}")

if 'epoch' in ckpt:
    epoch = ckpt['epoch']
    print(f"Epoch: {epoch}")

# 提取 model 键下的 state_dict
if 'model' in ckpt:
    state_dict = ckpt['model']
elif 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    # 如果没有 model 或 state_dict 键，假设整个检查点就是 state_dict
    state_dict = ckpt
    print("\n警告: 未找到 'model' 或 'state_dict' 键，使用整个检查点作为 state_dict")

print(f"\n=== Model 参数信息 ===")
print(f"总参数数量: {len(state_dict)}")
print(f"总参数量 (元素总数): {sum(v.numel() for v in state_dict.values())}")

print("\n=== 模型主要模块 (第一层前缀) ===")
# 提取所有 key 的第一部分 (例如 'detector.tracker.sam_mask_decoder...' -> 'detector')
prefixes = set()
for key in state_dict.keys():
    # 以 "." 分割，取第一个词
    prefix = key.split('.')[0]
    prefixes.add(prefix)

for p in sorted(list(prefixes)):
    # 统计该前缀下的参数数量
    count = sum(1 for k in state_dict.keys() if k.startswith(p + '.'))
    # 计算该前缀下的总参数量
    param_count = sum(v.numel() for k, v in state_dict.items() if k.startswith(p + '.'))
    print(f"- {p} ({count} 个参数, 参数量: {param_count})")

# print("\n=== 详细层级 (前两层前缀) ===")
# # 如果第一层只有一个前缀，那可能不够看，我们看看前两层
# prefixes_depth2 = set()
# for key in state_dict.keys():
#     parts = key.split('.')
#     if len(parts) > 1:
#         prefix = f"{parts[0]}.{parts[1]}"
#         prefixes_depth2.add(prefix)

# for p in sorted(list(prefixes_depth2)):
#     # 统计该前缀下的参数数量
#     count = sum(1 for k in state_dict.keys() if k.startswith(p + '.'))
#     print(f"- {p} ({count} 个参数)")

# print("\n=== 前10个参数键示例 ===")
# for i, key in enumerate(list(state_dict.keys())[:10]):
#     print(f"{i+1}. {key}")

