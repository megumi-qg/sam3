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
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path= "/home/gaoqi/sam3/experiment/acdc_finetune/checkpoints/checkpoint_converted.pt"

model= build_sam3_image_model(
    bpe_path=bpe_path,
    device=device,
    eval_mode=True,
    checkpoint_path=checkpoint_path,
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)

# - Load image
print("Loading image ...")
image_path= "/home/gaoqi/sam3/dataset/ACDC_new/test/images/patient102_frame01_slice003.png"
# image = Image.open(image_path).convert('RGB')
image = Image.open(image_path)
width, height = image.size
print(f"Image width={width}, height={height}")

# - Transform image
print("Transforming image ...")
resize_size= 1008
processor = Sam3Processor(model, resolution=resize_size, confidence_threshold=0.0)
inference_state = processor.set_image(image) ## Looking at the code, image is resized inside set_image method

# - Inference
prompt= "right ventricle"
processor.reset_all_prompts(inference_state)
inference_state= processor.set_text_prompt(state=inference_state, prompt=prompt)

masks, boxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]
import pdb; pdb.set_trace()
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