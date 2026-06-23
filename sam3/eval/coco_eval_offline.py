# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
This evaluator is meant for regular COCO mAP evaluation, for example on the COCO val set.

For Category mAP, we need the model to make predictions for all the categories on every single image.
In general, since the number of classes can be big, and the API model makes predictions individually for each pair (image, class),
we may need to split the inference process for a given image in several chunks.
"""

import logging
import contextlib
import io
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.train.utils.distributed import is_main_process

try:
    from tidecv import datasets, TIDE

    HAS_TIDE = True
except ImportError:
    HAS_TIDE = False
    print("WARNING: TIDE not installed. Detailed analysis will not be available.")


# the COCO detection metrics (https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L460-L471)
COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def convert_to_xywh(boxes):
    """Convert bounding boxes from xyxy format to xywh format."""
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


class HeapElement:
    """Utility class to make a heap with a custom comparator"""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class COCOevalCustom(COCOeval):
    """
    This is a slightly modified version of the original COCO API with added support for positive split evaluation.
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="segm", dt_only_positive=False
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.dt_only_positive = dt_only_positive

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        _gts_cat_ids = defaultdict(set)  # gt for evaluation on positive split
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
            _gts_cat_ids[gt["image_id"]].add(gt["category_id"])

        #### BEGIN MODIFICATION ####
        for dt in dts:
            if (
                self.dt_only_positive
                and dt["category_id"] not in _gts_cat_ids[dt["image_id"]]
            ):
                continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        #### END MODIFICATION ####
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results


class CocoEvaluatorOfflineWithPredFileEvaluators:
    def __init__(
        self,
        gt_path,
        tide: bool = True,
        iou_type: str = "bbox",
        positive_split=False,
        output_metrics: Optional[list[str]] = None,
        print_summary: bool = True,
        output_dice: bool = False,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.positive_split = positive_split
        self.iou_type = iou_type
        self.output_metrics = output_metrics
        self.print_summary = print_summary
        self.output_dice = output_dice

    def _compute_mean_dice(self, coco_dt):
        gt_by_img_cat = defaultdict(list)
        for ann in self.gt.dataset.get("annotations", []):
            gt_by_img_cat[(ann["image_id"], ann["category_id"])].append(ann)

        dt_by_img_cat = defaultdict(list)
        for ann in coco_dt.dataset.get("annotations", []):
            dt_by_img_cat[(ann["image_id"], ann["category_id"])].append(ann)

        dice_scores = []
        for key, gt_anns in gt_by_img_cat.items():
            image_id, category_id = key
            image_info = self.gt.imgs[image_id]
            height = image_info["height"]
            width = image_info["width"]

            gt_mask = np.zeros((height, width), dtype=bool)
            for ann in gt_anns:
                gt_mask |= self.gt.annToMask(ann).astype(bool)

            pred_mask = np.zeros((height, width), dtype=bool)
            dt_anns = dt_by_img_cat.get((image_id, category_id), [])
            if dt_anns:
                best_dt = max(dt_anns, key=lambda ann: ann.get("score", 0.0))
                pred_mask = coco_dt.annToMask(best_dt).astype(bool)

            gt_area = gt_mask.sum()
            pred_area = pred_mask.sum()
            if gt_area == 0:
                continue
            if pred_area == 0:
                dice_scores.append(0.0)
                continue
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            dice_scores.append((2.0 * intersection) / (gt_area + pred_area))

        if not dice_scores:
            return 0.0
        return float(np.mean(dice_scores))

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        self.gt = COCO(self.gt_path)

        # Creating the result file
        logging.info("Coco evaluator: Creating the result file")
        cocoDt = self.gt.loadRes(str(dumped_file))

        # Run the evaluation
        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOevalCustom(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        if self.print_summary:
            coco_eval.summarize()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            metric_name = COCO_METRICS[i]
            if self.output_metrics is not None and metric_name not in self.output_metrics:
                continue
            outs[f"coco_eval_{self.iou_type}_{metric_name}"] = value

        if self.output_dice and self.iou_type == "segm":
            outs[f"coco_eval_{self.iou_type}_Dice"] = self._compute_mean_dice(cocoDt)

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            self.tide_gt = datasets.COCO(self.gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")

            # Run TIDE
            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

        return outs
