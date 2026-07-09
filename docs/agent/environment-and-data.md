# Environment And Data

## Environment

Default data root:

- `/home/gaoqi/dataset/using`

Default Conda environment:

- `sam3`

Before running training, evaluation, or preprocessing scripts, use:

```bash
source /home/gaoqi/anaconda3/etc/profile.d/conda.sh
conda activate sam3
```

Unless the user says otherwise, use each dataset's `processed` directory, not `raw`.

## Datasets

Current important datasets:

- ACDC
- MSCMR
- ISBI

## ACDC

Default 2D processed path:

- `/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100`

Default 3D/video-like processed path:

- `/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100`

ACDC is the main reference dataset for local pipeline behavior.

## MSCMR

Default 2D processed path:

- `/home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes`

Default video-like path used by current joint tracker configs:

- `/home/gaoqi/dataset/using/mscmr/processed/sam3_video_npz_coco_fullframes`

Important split semantics:

- `raw/train` currently has scribble labels, not complete dense/manual labels.
- `raw/val` and `raw/test` have complete labels.
- If `train/full_annotations.coco.json` is used for MSCMR training, first confirm where its full labels came from.

Do not assume MSCMR has the same full train/val/test dense-label structure as ACDC.

## ISBI

Previous 2D path:

- `/home/gaoqi/dataset/using/isbi/processed/png_coco_sam3_fullframes`

Current video-like path used by joint tracker configs:

- `/home/gaoqi/dataset/using/isbi/processed/sam3_video_npz_coco_fullframes_train_val_test`

Important metric caveat:

- Existing H5-to-NIfTI data may not contain trustworthy physical spacing.
- If no explicit spacing file is used, treat Dice/IoU as reliable and treat HD95/NSD as potentially meaningless or `nan`.

## Evaluation Metrics

Common metrics:

- Dice
- IoU
- HD95
- NSD

HD95 and NSD depend on spacing. Evaluation should prefer `spacing_map.json` when available.
