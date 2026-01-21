import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.eval.postprocessors import PostProcessImage

import torch
import torchvision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#########################
##   MAIN
#########################
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
print("sam3_root")
print(sam3_root)
device = "cuda" if torch.cuda.is_available() else "cpu"

# - Build model
print("Loading model ...")
bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path= "/home/gaoqi/sam3/experiment/acdc_camus_weak_scribble/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt"

# 预处理检查点：处理 model 键和 detector 前缀
print(f"预处理检查点文件: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# 提取 model 键（如果存在）
if "model" in ckpt and isinstance(ckpt["model"], dict):
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

# 检查键是否包含 "detector" 前缀
keys = list(state_dict.keys())
has_detector_prefix = any("detector" in k for k in keys)

if not has_detector_prefix:
    print("检测到键没有 'detector.' 前缀，自动添加...")
    from collections import OrderedDict
    processed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = "detector." + k
        processed_state_dict[new_key] = v
    state_dict = processed_state_dict
    print("已添加 'detector.' 前缀")
else:
    print("检查点已包含 'detector.' 前缀，无需处理")

# 构建模型（不加载检查点）
model= build_sam3_image_model(
    bpe_path=bpe_path,
    device=device,
    eval_mode=True,
    checkpoint_path=None,  # 不通过文件路径加载
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)

# 手动加载处理后的检查点
print("加载检查点到模型...")
# 按照 _load_checkpoint 的逻辑处理检查点
sam3_image_ckpt = {
    k.replace("detector.", ""): v for k, v in state_dict.items() if "detector" in k
}
if model.inst_interactive_predictor is not None:
    sam3_image_ckpt.update(
        {
            k.replace("tracker.", "inst_interactive_predictor.model."): v
            for k, v in state_dict.items()
            if "tracker" in k
        }
    )
missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
if len(missing_keys) > 0:
    print(f"加载检查点时发现缺失的键: {missing_keys}")
else:
    print("检查点加载成功！")

# - Load image
print("Loading image ...")
image_path= "/home/gaoqi/dataset/using/acdc3/test/images/patient103_frame01_slice003.png"
# image = Image.open(image_path).convert('RGB')
image = Image.open(image_path)
width, height = image.size
print(f"Image width={width}, height={height}")

# - Transform image
print("Transforming image ...")
resize_size= 1008
processor = Sam3Processor(model, resolution=resize_size, confidence_threshold=0.1)
inference_state = processor.set_image(image) ## Looking at the code, image is resized inside set_image method

# - Inference
prompt= "right ventricle"
processor.reset_all_prompts(inference_state)
inference_state= processor.set_text_prompt(state=inference_state, prompt=prompt)

masks, boxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]

# - Draw results
img0 = Image.open(image_path)
plot_results(img0, inference_state)


# Save result image to disk (remote server has no display)
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
# use image basename to make filename informative
output_path = os.path.join(output_dir, f"inference_result_{os.path.basename(image_path)}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved result image to: {output_path}")