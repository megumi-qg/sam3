import sys
import os
sys.path.append('/home/gaoqi/sam3')

import torch
from torchinfo import summary
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import BatchedDatapoint, FindStage, BatchedFindTarget, BatchedInferenceMetadata
# /home/gaoqi/official_ckpt/sam3_hf/sam3.pt
# /home/gaoqi/sam3/gq_experiment/acdc_camus/weak/scribble/checkpoints/val_acdc_segmentation_coco_eval_segm_AP_model_only.pt
# 加载保存的 model state_dict
MODEL_PATH = '/home/gaoqi/sam3/gq_experiment/acdc_camus/weak/scribble/checkpoints/val_acdc_segmentation_coco_eval_segm_AP_model_only.pt'

print(f"正在加载模型: {MODEL_PATH}")
model_state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

# 构建 SAM3 图像模型实例（不加载检查点）
model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False)

# 加载 state_dict 到模型中
model.load_state_dict(model_state_dict, strict=False)

# 设置为评估模式
model.eval()

print("模型加载成功！")
print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

# 创建示例输入数据
# input_data = BatchedDatapoint(
#     img_batch=torch.randn(1, 3, 1008, 1008),
#     find_text_batch=[""],
#     find_inputs=[
#         FindStage(
#             img_ids=[0],
#             text_ids=[0],
#             input_boxes=torch.randn(1, 4),
#             input_boxes_label=torch.randn(1, 1),
#             input_boxes_mask=torch.randn(1),
#             input_points=torch.randn(1, 2),
#             input_points_mask=torch.randn(1),
#             object_ids=[0],
#         )
#     ],
#     find_targets=[
#         BatchedFindTarget(
#             num_boxes=[1],
#             boxes=torch.randn(1, 4),
#             boxes_padded=torch.randn(1, 1, 4),
#             is_exhaustive=[True],
#             segments=[torch.randn(1, 1008, 1008)],
#             semantic_segments=[torch.randn(1008, 1008)],
#             is_valid_segment=[True],
#             repeated_boxes=torch.randn(1, 4),
#             object_ids=[0],
#             object_ids_padded=torch.randn(1, 1),
#         )
#     ],
#     find_metadatas=[
#         BatchedInferenceMetadata(
#             coco_image_id=[0],
#             original_size=[(1008, 1008)],
#             object_id=[0],
#             frame_index=[0],
#             original_image_id=[0],
#             original_category_id=[0],
#             is_conditioning_only=[False],
#         )
#     ],
# )

# print("\n=== 模型摘要 ===")
# try:
#     summary(model, input_data=input_data)
# except Exception as e:
#     print(f"使用示例输入失败: {e}")
#     print("尝试手动计算参数...")
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"总参数: {total_params}")
#     print(f"可训练参数: {trainable_params}")
    
#     print("\n=== 模型结构 ===")
#     def print_model_structure(module, name='', indent=0):
#         print('  ' * indent + f"{name} ({type(module).__name__})")
#         for child_name, child_module in module.named_children():
#             print_model_structure(child_module, child_name, indent + 1)
    
#     print_model_structure(model)