# Video-Like Data And Slice Context

## Video-Like 3D Path

The repository supports organizing a 3D volume as a video-like sample.

Key files:

- `sam3/train/data/coco_json_loaders.py`
- `sam3/train/data/sam3_video_dataset.py`
- `sam3/model/sam3_image.py`
- `sam3/model_builder.py`

Current behavior:

- `VideoGroundingDataset` and `COCO_VIDEO_FROM_JSON` support volume-as-sample data.
- Images can be loaded slice-by-slice from `.npz`.
- `Sam3ImageOnVideoMultiGPU` can receive multi-frame inputs.
- The forward path still mainly runs `forward_grounding` frame-by-frame.

So the current "3D input" is multi-slice packaging, not full cross-slice attention, memory, or feature fusion.

## Batch Semantics

Do not interpret `train_batch_size=2` in a 3D config as equivalent to old 2D `batch_size=18`.

More accurate interpretation:

- `train_batch_size=2` means two 3D volume samples per step.
- Training usually samples only part of each volume.
- If `num_stages_sample=4`, the effective per-step slice count is closer to `2 x 4 = 8`.
- Video-like data has extra stage/query organization overhead, so memory behavior differs from pure 2D training.

## Video Weak Supervision

Video annotation JSON can carry `valid_mask`.

Rules:

- 3D scribble video JSON reuses the same `1/0/255` semantics as 2D weak supervision.
- Full masks remain `0/1`.

## Context V1

Context v1 is implemented as neighbor-slice feature prompting.

Key files:

- `sam3/model/slice_context_adapter.py`
- `sam3/model/sam3_image_slice_context.py`
- `sam3/model_builder.py`
- `gq_scripts/evaluate/batch_inference_context.py`
- `gq_scripts/evaluate/run_context_inference_and_eval.sh`

Configs:

- `sam3/train/configs/acdc/full_video_lora_100_context_v1.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100_context_v1.yaml`

Mechanism:

- Input is a continuous window, previously `window size = 5`.
- The center slice is the prediction target.
- Neighbor slices pass through the backbone.
- Neighbor features are compressed with `AdaptiveAvgPool2d((2,2)) + Linear`.
- Each neighbor slice produces 4 tokens.
- Four neighbor slices produce `16 x 256` visual prompt tokens.
- Relative slice-position embeddings are added.
- Tokens are injected as `visual_prompt_embed` into `_encode_prompt(...)`.

This is not a true 3D encoder.

## Context V1 Result

ACDC context v1 was fully connected through training, validation, test inference, and evaluation, but it underperformed the no-context baseline:

- Full baseline Dice: `0.9323`
- Full context v1 Dice: `0.9063`
- Scribble baseline Dice: `0.9130`
- Scribble context v1 Dice: `0.8572`

Current conclusion:

- Context v1 is an engineering baseline and negative result.
- Direct neighbor-feature prompt injection appears too disruptive for the current image-model path.
- Future context work should start from conservative feature-path fusion.

## Context V2 Direction

Preferred direction:

- Use a minimal neighborhood such as `center±1`.
- Use gated residual fusion.
- Initialize gates near zero.
- Fuse into the image feature path instead of forcing context through prompt tokens.
- First target "does not hurt baseline" before seeking a large gain.
