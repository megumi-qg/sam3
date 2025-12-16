import torch

# 加载文件
ckpt = torch.load('/home/gaoqi/sam3/experiment/acdc_finetune/checkpoints/checkpoint.pt', map_location='cpu')

# 这里不需要 if 'model' in ckpt 了，因为我们已经确认它就是参数字典
state_dict = ckpt

print(f"总参数数量: {len(state_dict)}")
print("\n=== 模型主要模块 (第一层前缀) ===")

# 提取所有 key 的第一部分 (例如 'tracker.sam_mask_decoder...' -> 'tracker')
prefixes = set()
for key in state_dict.keys():
    # 以 "." 分割，取第一个词
    prefix = key.split('.')[0]
    prefixes.add(prefix)

for p in sorted(list(prefixes)):
    print(f"- {p}")

print("\n=== 详细层级 (前两层前缀) ===")
# 如果第一层只有一个 'tracker'，那可能不够看，我们看看前两层
prefixes_depth2 = set()
for key in state_dict.keys():
    parts = key.split('.')
    if len(parts) > 1:
        prefix = f"{parts[0]}.{parts[1]}"
        prefixes_depth2.add(prefix)

for p in sorted(list(prefixes_depth2)):
    print(f"- {p}")