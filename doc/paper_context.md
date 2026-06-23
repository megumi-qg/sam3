# Paper Context

## Files

- MICCAI draft: `gq_paper/miccai/paper_latex/`
- MICCAI review comments: `gq_paper/miccai/审稿意见.md`
- CMPB Elsevier draft: `gq_paper/cmpb/elsarticle/main.tex`
- CMPB supplementary draft: `gq_paper/cmpb/elsarticle/supplementary.tex`
- CMPB highlights: `gq_paper/cmpb/elsarticle/highlights.tex`
- CMPB submission notes: `gq_paper/cmpb/elsarticle/CMPB_SUBMISSION_NOTES.md`
- Results aggregation: `gq_paper/cmpb/aggregate_cmpb_results.py`
- Result summaries: `gq_paper/cmpb/results_summary/`

## Manuscript State

The CMPB manuscript has been migrated from MICCAI LNCS format to Elsevier
`elsarticle`.

Current status:

- Introduction has been rewritten.
- Related Work has been split into an independent section.
- The main narrative has moved away from strong "interactive segmentation"
  wording toward "text-prompted medical image segmentation".
- Main tables use latest aggregated model results.
- Per-structure tables are in standalone supplementary material.
- O2 no geometric matcher has been adopted as the proposed SAM3-Scribble
  result.

## Compilation

Compile from `gq_paper/cmpb/elsarticle/` inside the `sam3` conda environment:

```bash
tectonic main.tex
tectonic supplementary.tex
```

## CMPB Positioning

CMPB fits biomedical computing methodology and software systems in biomedical
research and medical practice. The paper should be positioned as a complete,
reproducible, and sufficiently validated medical computing method, not merely
as a model-running report.
