# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
from pycocotools import mask as mask_util

FRAME_ID_STRIDE = 1_000_000


# ============================================================================
# Utility Functions
# ============================================================================


def convert_boxlist_to_normalized_tensor(box_list, image_width, image_height):
    """
    Converts a list of bounding boxes to a normalized PyTorch tensor.

    Args:
        box_list (list of list or tuples): Each box is [x_min, y_min, x_max, y_max].
        image_width (int or float): Width of the image.
        image_height (int or float): Height of the image.

    Returns:
        torch.Tensor: Normalized tensor of shape (N, 4), values in [0, 1].
    """
    boxes = torch.tensor(box_list, dtype=torch.float32)
    boxes[:, [0, 2]] /= image_width  # x_min, x_max
    boxes[:, [1, 3]] /= image_height  # y_min, y_max
    boxes = boxes.clamp(0, 1)
    return boxes


def load_coco_and_group_by_image(json_path: str) -> Tuple[List[Dict], Dict[int, Union[str, List[str]]]]:
    """
    Load COCO JSON file and group annotations by image.

    Args:
        json_path (str): Path to COCO JSON file.

    Returns:
        Tuple containing:
            - List of dicts with 'image' and 'annotations' keys
            - Dict mapping category IDs to category names (str) or list of names (List[str])
              If a category has "names" field (list), it will be used; otherwise "name" field (str) will be used.
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    sorted_image_ids = sorted(images.keys())

    grouped = []
    for image_id in sorted_image_ids:
        image_info = images[image_id]
        grouped.append(
            {"image": image_info, "annotations": anns_by_image.get(image_id, [])}
        )

    cat_id_to_name = {}
    for cat in coco["categories"]:
        cat_id = cat["id"]
        if "names" in cat and isinstance(cat["names"], list):
            cat_id_to_name[cat_id] = cat["names"]
        elif "name" in cat:
            cat_id_to_name[cat_id] = cat["name"]
        else:
            raise ValueError(f"Category {cat_id} must have either 'name' or 'names' field")

    return grouped, cat_id_to_name


def ann_to_rle(segm, im_info: Dict) -> Dict:
    """
    Convert annotation which can be polygons or uncompressed RLE to RLE.

    Args:
        segm: Segmentation data (polygon list or RLE dict)
        im_info (dict): Image info containing 'height' and 'width'

    Returns:
        RLE encoded segmentation
    """
    h, w = im_info["height"], im_info["width"]

    if isinstance(segm, list):
        # Polygon - merge all parts into one mask RLE code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm["counts"], list):
        # Uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # Already RLE
        rle = segm

    return rle


def make_video_frame_image_id(video_id: int, frame_idx: int) -> int:
    """
    Build a globally unique frame id from (video_id, frame_idx).

    Keep this deterministic mapping aligned with the 3D video preprocessing so
    frame-level validation predictions can be matched with COCO-style GT.
    """
    return int(video_id) * FRAME_ID_STRIDE + int(frame_idx)


# ============================================================================
# COCO Training API
# ============================================================================



class COCO_FROM_JSON:
    """
    COCO training API for loading box-only annotations from JSON.
    Groups all annotations per image and creates queries per category.
    """

    def __init__(
        self,
        annotation_file,
        prompts=None,
        include_negatives=True, # 空数据非常重要，构成了负样本
        category_chunk_size=None,
    ):
        """
        Initialize the COCO training API.

        Args:
            annotation_file (str): Path to COCO JSON annotation file
            prompts: Optional custom prompts for categories
            include_negatives (bool): Whether to include negative examples (categories with no instances)
        """
        # self._raw_data按照图像对annotation分组
        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(annotation_file)
        self._sorted_cat_ids = sorted(list(self._cat_idx_to_text.keys()))
        self.prompts = None
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size
            if category_chunk_size is not None
            else len(self._sorted_cat_ids)
        )
        # 举例，假如有9个类别，category_chunk_size=3，则self.category_chunks为：
        # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        if prompts is not None:
            prompts = eval(prompts)
            self.prompts = {}
            for loc_dict in prompts:
                cat_id = int(loc_dict["id"])
                if "names" in loc_dict and isinstance(loc_dict["names"], list):
                    self.prompts[cat_id] = loc_dict["names"]
                elif "name" in loc_dict:
                    self.prompts[cat_id] = loc_dict["name"]
                else:
                    raise ValueError(f"Prompt for category {cat_id} must have either 'name' or 'names' field")
            assert len(self.prompts) == len(
                self._sorted_cat_ids
            ), "Number of prompts must match number of categories"

    def getDatapointIds(self):
        """Return all datapoint indices for training."""
        return list(range(len(self._raw_data) * len(self.category_chunks)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]

        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,  # Single image per datapoint
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        annot_template = {
            "image_id": 0,
            "bbox": None,  # Normalized bbox in xywh
            "area": None,  # json文件中，指像素点的个数；如果提供bbox, 会计算为归一化的bbox的面积
            "segmentation": None,  # RLE encoded
            "valid_mask": None,
            "object_id": None,
            "is_crowd": None,
            "id": None,
        }

        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]
        # Group annotations by category
        cat_id_to_anns = defaultdict(list)
        for ann in raw_annotations:
            cat_id_to_anns[ann["category_id"]].append(ann)

        annotations_by_cat_sorted = [
            (cat_id, cat_id_to_anns[cat_id]) for cat_id in cat_chunk
        ]

        for cat_id, anns in annotations_by_cat_sorted:
            if len(anns) == 0 and not self.include_negatives:
                continue

            cur_ann_ids = []
            
            # Create annotations for this category
            for ann in anns:
                annotation = annot_template.copy()
                annotation["id"] = len(annotations)
                annotation["object_id"] = annotation["id"]
                annotation["is_crowd"] = ann["iscrowd"]

                # Missing bbox: placeholder; dataset may recover from segmentation
                if "bbox" in ann:
                    normalized_boxes = convert_boxlist_to_normalized_tensor(
                        [ann["bbox"]], width, height
                    )
                    bbox = normalized_boxes[0]
                    annotation["area"] = (bbox[2] * bbox[3]).item()
                    annotation["bbox"] = bbox
                else:
                    annotation["bbox"] = torch.tensor([0.0, 0.0, 0.0, 0.0])
                    if "area" in ann:
                        annotation["area"] = float(ann["area"])
                    else:
                        annotation["area"] = 0.0
                if (
                    "segmentation" in ann
                    and ann["segmentation"] is not None
                    and ann["segmentation"] != []
                ):
                    annotation["segmentation"] = ann_to_rle(
                        ann["segmentation"], im_info=image_info
                    )
                if (
                    "valid_mask" in ann
                    and ann["valid_mask"] is not None
                    and ann["valid_mask"] != []
                ):
                    annotation["valid_mask"] = ann_to_rle(
                        ann["valid_mask"], im_info=image_info
                    )

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])

            # Create query for this category
            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            
            if self.prompts is not None:
                prompt_value = self.prompts[cat_id]
            else:
                prompt_value = self._cat_idx_to_text[cat_id]

            if isinstance(prompt_value, list):
                query["query_text"] = random.choice(prompt_value)
            else:
                query["query_text"] = prompt_value
                        
            query["object_ids_output"] = cur_ann_ids
            queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dict
        """
        img_idx = idx // len(self.category_chunks)
        img_data = self._raw_data[img_idx]["image"]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class COCO_VIDEO_FROM_JSON:
    """COCO video training API for loading video annotations from JSON."""

    def __init__(
        self,
        annotation_file,
        prompts=None,
        include_negatives=True,
        category_chunk_size=None,
        max_frames_per_video=None,  # ← NEW PARAMETER
    ):
        with open(annotation_file, "r") as f:
            data = json.load(f)
        
        # Check if this is video format
        if "videos" in data:
            self._videos = {v["id"]: v for v in data["videos"]}
            # IMPORTANT: For video format, images are optional
            self._images = {img["id"]: img for img in data.get("images", [])}
        else:
            raise ValueError("JSON must contain 'videos' field for video training")
        
        # Group annotations by video_id
        self._anns_by_video = defaultdict(list)
        for ann in data["annotations"]:
            self._anns_by_video[ann["video_id"]].append(ann)
        
        # Categories
        self._cat_idx_to_text = {cat["id"]: cat["name"] for cat in data["categories"]}
        self._sorted_cat_ids = sorted(list(self._cat_idx_to_text.keys()))
        
        self.prompts = None
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size
            if category_chunk_size is not None
            else len(self._sorted_cat_ids)
        )
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        
        if prompts is not None:
            prompts = eval(prompts)
            self.prompts = {}
            for loc_dict in prompts:
                self.prompts[int(loc_dict["id"])] = loc_dict["name"]


    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for video datapoint.
        Handles both regular videos and NPZ medical imaging data.
        """
        video_idx = idx // len(self.category_chunks)
        video_id_list = sorted(self._videos.keys())
        
        if video_idx >= len(video_id_list):
            return []
        
        video_id = video_id_list[video_idx]
        video_info = self._videos[video_id]
        num_frames = video_info.get("length", 0) # 这里的 frames 是 3D 的 slices
        
        images = []
        
        # ✅ For NPZ medical data
        if "npz_path" in video_info and video_info["npz_path"]:
            npz_path = video_info["npz_path"]

            # 这里我们并不真正加载庞大的 3D 数据，只是生成元数据
            # 实际的数据加载通常发生在 dataset 的 __getitem__ 里
            for frame_idx in range(num_frames):
                images.append({
                    "id": frame_idx,
                    "file_name": npz_path,  # ✅ Use NPZ path as file_name!
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "original_img_id": make_video_frame_image_id(video_id, frame_idx),
                    "coco_img_id": make_video_frame_image_id(video_id, frame_idx),
                    "height": video_info["height"],
                    "width": video_info["width"],
                    "is_npz": True,  # ✅ Add explicit flag
                })
        
        # For regular video files
        elif "file_names" in video_info:
            for frame_idx, fname in enumerate(video_info["file_names"]):
                images.append({
                    "id": frame_idx,
                    "file_name": fname,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "original_img_id": make_video_frame_image_id(video_id, frame_idx),
                    "coco_img_id": make_video_frame_image_id(video_id, frame_idx),
                    "height": video_info["height"],
                    "width": video_info["width"],
                    "is_npz": False,
                })
        
        return images

    def getDatapointIds(self):
        """Return all datapoint indices for training."""
        return list(range(len(self._videos) * len(self.category_chunks)))

    def loadImgs(self, ids):
        """
        Load video/image metadata for given IDs.
        
        CRITICAL: For NPZ video data, we return video metadata with NPZ path.
        """
        if not hasattr(ids, '__iter__'):
            ids = [ids]
        
        results = []
        for datapoint_id in ids:
            # Map datapoint_id to video_id
            video_idx = datapoint_id // len(self.category_chunks)
            video_id_list = sorted(self._videos.keys())
            
            if video_idx >= len(video_id_list):
                continue
            
            video_id = video_id_list[video_idx]
            video_info = self._videos[video_id]
            
            # Return video metadata with NPZ information
            result = {
                'id': datapoint_id,  # Keep original datapoint ID
                'video_id': video_id,
                'file_name': video_info.get('file_names', [''])[0],  # First frame
                'video_name': video_info.get('video_name'),
                'npz_path': video_info.get('npz_path'),  # CRITICAL for NPZ loading
                'slice_indices': video_info.get('slice_indices'),  # CRITICAL for NPZ loading
                'height': video_info['height'],
                'width': video_info['width'],
            }
            results.append(result)
        
        return results

    def loadQueriesAndAnnotationsFromDatapoint(self, idx, max_frames=None):
        """Load queries and annotations - DEBUG VERSION."""
        video_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]
        
        video_id_list = sorted(self._videos.keys())
        if video_idx >= len(video_id_list):
            return [], []
        
        video_id = video_id_list[video_idx]
        video_info = self._videos[video_id]
        
        queries = []
        annotations = []
        annotation_id_counter = 0  # ← Global counter
        
        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": None,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": None,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        
        annot_template = {
            "image_id": None,
            "bbox": None,
            "area": None,
            "segmentation": None,
            "seed_segmentation": None,
            "valid_mask": None,
            "object_id": None,
            "is_crowd": 0,
            "id": None,
        }
        
        raw_annotations = self._anns_by_video.get(video_id, [])
        
        cat_id_to_anns = defaultdict(list)
        for ann in raw_annotations:
            cat_id_to_anns[ann["category_id"]].append(ann)
        
        width, height = video_info["width"], video_info["height"]
        num_frames = video_info["length"]

        # if max_frames is not None:
        #     num_frames = min(num_frames, max_frames)

        #     num_frames = video_info["length"]

        # Only limit if explicitly requested
        if max_frames is not None and max_frames > 0:
            print(f"⚠️  WARNING: Limiting frames from {num_frames} to {max_frames}")
            num_frames = min(num_frames, max_frames)


        # Build one query per (frame, category) if include_negatives=True.
        # This keeps the number of queries consistent across stages, which is
        # required later by the dataset and collator.
        for frame_idx in range(num_frames):
            for cat_id in cat_chunk:
                anns = cat_id_to_anns.get(cat_id, [])
                cur_ann_ids = []

                for obj_idx, obj_ann in enumerate(anns):
                    # Handle sparse format
                    if "frame_indices" in obj_ann:
                        frame_indices = obj_ann["frame_indices"]
                        
                        if frame_idx not in frame_indices:
                            continue
                        
                        list_idx = frame_indices.index(frame_idx)
                        
                        if list_idx >= len(obj_ann["bboxes"]):
                            continue
                        
                        bbox = obj_ann["bboxes"][list_idx]
                        segmentation = obj_ann["segmentations"][list_idx] if list_idx < len(obj_ann["segmentations"]) else []
                        seed_segmentation = (
                            obj_ann["seed_segmentations"][list_idx]
                            if "seed_segmentations" in obj_ann
                            and list_idx < len(obj_ann["seed_segmentations"])
                            else None
                        )
                        valid_mask = (
                            obj_ann["valid_masks"][list_idx]
                            if "valid_masks" in obj_ann and list_idx < len(obj_ann["valid_masks"])
                            else None
                        )
                    
                    # Handle dense format
                    elif "bboxes" in obj_ann and isinstance(obj_ann["bboxes"], list):
                        if frame_idx >= len(obj_ann["bboxes"]):
                            continue
                        
                        bbox = obj_ann["bboxes"][frame_idx]
                        segmentation = obj_ann["segmentations"][frame_idx] if "segmentations" in obj_ann and frame_idx < len(obj_ann["segmentations"]) else []
                        seed_segmentation = (
                            obj_ann["seed_segmentations"][frame_idx]
                            if "seed_segmentations" in obj_ann
                            and frame_idx < len(obj_ann["seed_segmentations"])
                            else None
                        )
                        valid_mask = (
                            obj_ann["valid_masks"][frame_idx]
                            if "valid_masks" in obj_ann and frame_idx < len(obj_ann["valid_masks"])
                            else None
                        )
                    
                    else:
                        continue
                    
                    # Skip empty bboxes
                    if not bbox or sum(bbox) == 0:
                        continue
                    
                    if len(bbox) != 4:
                        continue
                    
                    # x1, y1, x2, y2 = bbox
                    x, y, w, h = bbox
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Clip to bounds
                    x1 = max(0.0, min(float(x1), float(width)))
                    y1 = max(0.0, min(float(y1), float(height)))
                    x2 = max(0.0, min(float(x2), float(width)))
                    y2 = max(0.0, min(float(y2), float(height)))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    bbox_xyxy = [x1, y1, x2, y2]
                    
                    # Normalize
                    normalized_boxes = convert_boxlist_to_normalized_tensor(
                        [bbox_xyxy], width, height
                    )
                    normalized_bbox = normalized_boxes[0]
                    
                    box_width = normalized_bbox[2] - normalized_bbox[0]
                    box_height = normalized_bbox[3] - normalized_bbox[1]
                    
                    if box_width <= 0 or box_height <= 0:
                        continue
                    
                    # Create annotation
                    annotation = annot_template.copy()
                    annotation["id"] = annotation_id_counter 
                    annotation["object_id"] = obj_ann["id"]
                    annotation["is_crowd"] = obj_ann.get("iscrowd", 0)
                    annotation["image_id"] = frame_idx
                    annotation["bbox"] = normalized_bbox
                    annotation["area"] = (box_width * box_height).item()
                    
                    if segmentation and segmentation != []:
                        annotation["segmentation"] = segmentation
                    if seed_segmentation and seed_segmentation != []:
                        annotation["seed_segmentation"] = seed_segmentation
                    if valid_mask and valid_mask != []:
                        annotation["valid_mask"] = valid_mask
                    
                    annotations.append(annotation)
                    cur_ann_ids.append(annotation_id_counter)
                    annotation_id_counter += 1  # ← Increment

                if len(cur_ann_ids) == 0 and not self.include_negatives:
                    continue
                
                # Create query
                query = query_template.copy()
                query["id"] = len(queries)
                query["original_cat_id"] = cat_id
                query["query_text"] = self._cat_idx_to_text[cat_id]
                query["object_ids_output"] = cur_ann_ids
                query["image_id"] = frame_idx
                query["query_processing_order"] = frame_idx
                queries.append(query)
    
        
        return queries, annotations



# ============================================================================
# SAM3 Evaluation APIs
# ============================================================================


class SAM3_EVAL_API_FROM_JSON_NP:
    """
    SAM3 evaluation API for loading noun phrase queries from JSON.
    """

    def __init__(self, annotation_file):
        """
        Initialize the SAM3 evaluation API.

        Args:
            annotation_file (str): Path to SAM3 JSON annotation file
        """
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self._image_data = data["images"]

    def getDatapointIds(self):
        """Return all datapoint indices."""
        return list(range(len(self._image_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        cur_img_data = self._image_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        # Create query
        query = query_template.copy()
        query["id"] = len(queries)
        query["original_cat_id"] = int(cur_img_data["queried_category"])
        query["query_text"] = cur_img_data["text_input"]
        if "is_instance_exhaustive" in cur_img_data:
            query["is_exhaustive"] = bool(cur_img_data["is_instance_exhaustive"])
        else:
            query["is_exhaustive"] = False
        query["object_ids_output"] = []
        queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dict
        """
        img_data = self._image_data[idx]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class SAM3_VEVAL_API_FROM_JSON_NP:
    """
    SAM3 video evaluation API for loading noun phrase queries from JSON.
    """

    def __init__(self, annotation_file):
        """
        Initialize the SAM3 video evaluation API.

        Args:
            annotation_file (str): Path to SAM3 video JSON annotation file
        """
        with open(annotation_file, "r") as f:
            data = json.load(f)

        assert "video_np_pairs" in data, "Incorrect data format"

        self._video_data = data["videos"]
        self._video_id_to_np_ids = defaultdict(list)
        self._cat_id_to_np = {}

        for cat_dict in data["categories"]:
            self._cat_id_to_np[cat_dict["id"]] = cat_dict["name"]

        for video_np_dict in data["video_np_pairs"]:
            self._video_id_to_np_ids[video_np_dict["video_id"]].append(
                video_np_dict["category_id"]
            )
            assert (
                self._cat_id_to_np[video_np_dict["category_id"]]
                == video_np_dict["noun_phrase"]
            ), "Category name does not match text input"

    def getDatapointIds(self):
        """Return all datapoint indices."""
        return list(range(len(self._video_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific video datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        cur_vid_data = self._video_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        all_np_ids = self._video_id_to_np_ids[cur_vid_data["id"]]

        for np_id in all_np_ids:
            text_input = self._cat_id_to_np[np_id]

            for i, image_path in enumerate(cur_vid_data["file_names"]):
                query = query_template.copy()
                query["id"] = len(queries)
                query["original_cat_id"] = np_id
                query["query_text"] = text_input
                query["image_id"] = i
                query["query_processing_order"] = i
                query["object_ids_output"] = []
                queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific video datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dicts for all frames
        """
        video_data = self._video_data[idx]
        images = [
            {
                "id": i,
                "file_name": file_name,
                "original_img_id": video_data["id"],
                "coco_img_id": video_data["id"],
            }
            for i, file_name in enumerate(video_data["file_names"])
        ]
        return images
