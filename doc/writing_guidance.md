# Writing Guidance

## Recommended Framing

Use:

- text-prompted segmentation
- text-prompted medical image segmentation
- weakly supervised SAM3 adaptation
- zero-click text-prompted workflow
- competitive with selected text-prompted baselines under our protocol
- approaches the fully supervised SAM3 counterpart

Avoid presenting the method as a standard iterative interactive segmentation
method.

## Safe Ablation Wording

Box/GIoU regression loss:

- Safe: "Removing scribble-derived box/GIoU regression losses yields the best
  overall Dice/NSD and avoids unreliable geometric supervision."
- Avoid: "Box/GIoU removal dramatically improves performance."

Matcher geometry cost:

- Safe: "Removing geometric matching costs gives a slightly better and simpler
  matching configuration; geometry costs do not provide a stable assignment
  benefit under sparse scribble supervision."
- Avoid overstating the gain, because O1/O2/O3 margins are small.

LoRA scope:

- Safe: "The proposed hybrid LoRA configuration is more robust than single-side
  adaptation. Vision-only adaptation remains competitive, whereas DETR-only
  adaptation substantially degrades performance."

LoRA rank:

- Safe: "Rank 8 is used as a balanced default; rank 4 is competitive, and rank
  16 does not improve performance."
- Avoid: "Rank 8 is strictly optimal."

Threshold:

- Safe: "A fixed confidence threshold of 0.7 provides a stable balance between
  false positives and missed detections for SAM3-Scribble."
- Avoid: "The threshold was tuned on the test set."

## Claims To Avoid

- Do not make "interactive segmentation" the core claim.
- Do not claim the method "outperforms fully supervised interactive baselines"
  unless the exact protocol supports it.
- Avoid informal words such as "surprisingly" in result interpretation.
- Avoid implying baseline unfairness without evidence.

## Data Leakage Wording

For BiomedParse / VoxTell or similar pretrained foundation baselines, use
conservative wording unless there is explicit evidence.

Safe:

- "possible overlap with large-scale pretraining corpora"
- "included for context and interpreted cautiously"

Avoid directly asserting data leakage or unfair comparison without a source.

## Known Weak Spot

BTCV small bowel is hard. It is acceptable to explain that weak BTCV Dice can be
reasonable while HD95/NSD are worse because the structure is complex and
discontinuous.
