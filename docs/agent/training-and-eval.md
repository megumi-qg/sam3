# Training And Evaluation

## Primary Configs

ACDC 2D image-model configs:

- `sam3/train/configs/acdc/full_lora_100.yaml`
- `sam3/train/configs/acdc/scribble_lora_100.yaml`

ACDC video-like image-model configs:

- `sam3/train/configs/acdc/full_video_lora_100.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100.yaml`

ACDC context configs:

- `sam3/train/configs/acdc/full_video_lora_100_context_v1.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100_context_v1.yaml`

Joint ACDC+MSCMR+ISBI image-model configs:

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_lora.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_lora_balanced.yaml`

Joint tracker configs:

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v3_temporal_area_centroid_val_macro.yaml`

## LoRA

LoRA implementation:

- `sam3/model/lora.py`

Model construction:

- `sam3/model_builder.py`

Current local convention:

- Most backbone components use LoRA.
- `mask_decoder`, `segmentation_head`, and `dot_prod_scoring` remain fully trainable.
- LoRA injection happens after loading the pretrained checkpoint.

When changing LoRA behavior, check:

- `sam3/model/lora.py`
- `sam3/model_builder.py`
- the active Hydra config
- inference-time LoRA merge/runtime path

If an image-model baseline suddenly degrades, first inspect LoRA loading/merge behavior before trusting the result.

## Inference And Evaluation Scripts

Common image-model scripts:

- `gq_scripts/evaluate/batch_inference.py`
- `gq_scripts/evaluate/batch_inference_context.py`
- `gq_scripts/evaluate/batch_evaluate.py`
- `gq_scripts/evaluate/batch_evaluate_utils.py`
- `gq_scripts/evaluate/run_inference_and_eval.sh`
- `gq_scripts/evaluate/run_context_inference_and_eval.sh`

Tracker scripts:

- `gq_scripts/evaluate/tracker_auto_seed_inference.py`
- `gq_scripts/evaluate/evaluate_tracker_coco_predictions.py`
- `gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh`

Current common inference threshold:

- `confidence_threshold=0.7`

Why:

- Some ACDC slices contain no target structure.
- A higher threshold reduces false positives on blank slices.

This threshold is a working convention, not a theory constant. If predictions collapse to nearly zero, lower thresholds such as `0.0` are useful for debugging.

## Final Metrics

For medical segmentation results, prefer:

- 3D Dice
- IoU
- HD95
- NSD

Do not treat 2D COCO AP as the final medical-performance metric.

## Command Convention

When giving training commands for long-running jobs, use `nohup` and redirect logs to a named file. Prefer this shape:

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/gaoqi/anaconda3/envs/sam3/bin/python sam3/train/train.py \
  -c configs/.../experiment.yaml \
  --use-cluster 0 \
  --num-gpus 1 \
  > experiment.log 2>&1 &
```

Short preprocessing, evaluation, inspection, and smoke-test commands do not need `nohup` unless they are expected to run for a long time.
