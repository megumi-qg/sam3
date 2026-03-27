# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import random  # gaoqi: 添加 random 模块，用于从多个 prompt 中随机选择
from collections import defaultdict
from typing import Dict, List, Tuple, Union  # gaoqi: 添加 Union 类型，支持 category 名称可以是字符串或字符串列表

import torch
from pycocotools import mask as mask_util


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
    # gaoqi: 修改返回类型，支持 category 名称可以是字符串（单个 prompt）或字符串列表（多个 prompt）
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

    # gaoqi: 支持读取单个 prompt（"name" 字段）或多个 prompt（"names" 字段），保持向后兼容
    cat_id_to_name = {}
    for cat in coco["categories"]:
        cat_id = cat["id"]
        if "names" in cat and isinstance(cat["names"], list):
            # gaoqi: 如果存在 "names" 列表，存储为列表（多个 prompt）
            cat_id_to_name[cat_id] = cat["names"]
        elif "name" in cat:
            # gaoqi: 如果只有 "name" 字符串，存储为字符串（单个 prompt，向后兼容）
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
                # gaoqi: 支持 prompts 参数中同时包含单个 prompt（"name"）或多个 prompt（"names"）
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
            "valid_mask": None,    # <--- gaoqi:【新增】添加 valid_mask 字段占位符
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

                # === 修改开始：处理 bbox 缺失的情况 ===
                # gaoqi: 如果 bbox 不存在，给予一个默认值，避免 Key Error
                # 后续 sam3_image_dataset.py 会检测并从 segmentation 重新计算
                if "bbox" in ann:
                    normalized_boxes = convert_boxlist_to_normalized_tensor(
                        [ann["bbox"]], width, height
                    )
                    bbox = normalized_boxes[0]
                    annotation["area"] = (bbox[2] * bbox[3]).item()
                    annotation["bbox"] = bbox
                else:
                    # 使用 0 填充，保持 tensor 格式
                    annotation["bbox"] = torch.tensor([0.0, 0.0, 0.0, 0.0])
                    # 尝试用 mask 计算 area，如果没有 mask 则为 0
                    if "area" in ann:
                        annotation["area"] = float(ann["area"])
                    else:
                        annotation["area"] = 0.0
                # === 修改结束 ===

                if (
                    "segmentation" in ann
                    and ann["segmentation"] is not None
                    and ann["segmentation"] != []
                ):
                    annotation["segmentation"] = ann_to_rle(
                        ann["segmentation"], im_info=image_info
                    )
                # === gaoqi:【新增】处理 valid_mask (有效区域约束) ===
                if (
                    "valid_mask" in ann
                    and ann["valid_mask"] is not None
                    and ann["valid_mask"] != []
                ):
                    annotation["valid_mask"] = ann_to_rle(
                        ann["valid_mask"], im_info=image_info
                    )
                # ==========================================

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])

            # Create query for this category
            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            
            # gaoqi: 获取当前 category 的 text prompt（可能是单个字符串或字符串列表）
            if self.prompts is not None:
                # gaoqi: 如果提供了自定义 prompts，使用自定义 prompts
                prompt_value = self.prompts[cat_id]
            else:
                # gaoqi: 否则使用 annotation 文件中的 prompts
                prompt_value = self._cat_idx_to_text[cat_id]
            
            # gaoqi: 如果 prompt_value 是列表（多个 prompt），随机选择一个；否则直接使用（单个 prompt）
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


        frame_cat_has_data = set()

        for frame_idx in range(num_frames):
            # print(f"\n  Checking frame {frame_idx}:")
            for cat_id in cat_chunk:
                # print(f"    Category {cat_id}:")
                anns = cat_id_to_anns.get(cat_id, [])
                # print(f"      Annotations for this cat: {len(anns)}")
                
                for obj_ann in anns:
                    # print(f"        Ann {obj_ann['id']}:")
                    # Check sparse format
                    if "frame_indices" in obj_ann:
                        # print(f"          frame_indices: {obj_ann['frame_indices']}")
                        if frame_idx in obj_ann["frame_indices"]:
                            # print(f"          ✅ Frame {frame_idx} FOUND! Adding to frame_cat_has_data")
                            frame_cat_has_data.add((frame_idx, cat_id))
                            break
                        # else:
                            # print(f"❌ Frame {frame_idx} NOT in frame_indices")
                    # Check dense format
                    elif "bboxes" in obj_ann:
                        print(f"          Dense format: checking bbox at index {frame_idx}")
                        if frame_idx < len(obj_ann["bboxes"]):
                            bbox = obj_ann["bboxes"][frame_idx]
                            print(f"          bbox: {bbox}")
                            if bbox and sum(bbox) != 0:
                                print(f"          ✅ Valid bbox! Adding to frame_cat_has_data")
                                frame_cat_has_data.add((frame_idx, cat_id))
                                break

        # print(f"\n✅ frame_cat_has_data: {frame_cat_has_data}")

        # ✅ STEP 2: Process only (frame, category) pairs that have data
        queries_created = 0
        queries_with_empty_anns = 0
        
        for frame_idx in range(num_frames):
            for cat_id in cat_chunk:
                # Skip if no data for this frame+category
                if (frame_idx, cat_id) not in frame_cat_has_data:
                    continue
                
                anns = cat_id_to_anns.get(cat_id, [])
                # frame_obj_idx = 0 
                cur_ann_ids = []
                
                # ✅ DEBUG: Track annotation processing for this query
                annotations_attempted = 0
                annotations_skipped = 0
                
                for obj_idx, obj_ann in enumerate(anns):
                    annotations_attempted += 1
                    
                    # Handle sparse format
                    if "frame_indices" in obj_ann:
                        frame_indices = obj_ann["frame_indices"]
                        
                        if frame_idx not in frame_indices:
                            annotations_skipped += 1
                            continue
                        
                        list_idx = frame_indices.index(frame_idx)
                        
                        if list_idx >= len(obj_ann["bboxes"]):
                            annotations_skipped += 1
                            continue
                        
                        bbox = obj_ann["bboxes"][list_idx]
                        segmentation = obj_ann["segmentations"][list_idx] if list_idx < len(obj_ann["segmentations"]) else []
                    
                    # Handle dense format
                    elif "bboxes" in obj_ann and isinstance(obj_ann["bboxes"], list):
                        if frame_idx >= len(obj_ann["bboxes"]):
                            annotations_skipped += 1
                            continue
                        
                        bbox = obj_ann["bboxes"][frame_idx]
                        segmentation = obj_ann["segmentations"][frame_idx] if "segmentations" in obj_ann and frame_idx < len(obj_ann["segmentations"]) else []
                    
                    else:
                        annotations_skipped += 1
                        continue
                    
                    # Skip empty bboxes
                    if not bbox or sum(bbox) == 0:
                        annotations_skipped += 1
                        continue
                    
                    if len(bbox) != 4:
                        annotations_skipped += 1
                        continue
                    
                    # x1, y1, x2, y2 = bbox
                    x, y, w, h = bbox
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    

                    
                    
                    if x2 <= x1 or y2 <= y1:
                        annotations_skipped += 1
                        continue
                    
                    # Clip to bounds
                    x1 = max(0.0, min(float(x1), float(width)))
                    y1 = max(0.0, min(float(y1), float(height)))
                    x2 = max(0.0, min(float(x2), float(width)))
                    y2 = max(0.0, min(float(y2), float(height)))
                    
                    if x2 <= x1 or y2 <= y1:
                        annotations_skipped += 1
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
                        annotations_skipped += 1
                        continue
                    
                    # Create annotation
                    annotation = annot_template.copy()
                    # annotation["id"] = len(annotations)
                    annotation["id"] = annotation_id_counter 
                    annotation["object_id"] = obj_ann["id"]
                    annotation["is_crowd"] = obj_ann.get("iscrowd", 0)
                    annotation["image_id"] = frame_idx
                    annotation["bbox"] = normalized_bbox
                    annotation["area"] = (box_width * box_height).item()
                    
                    if segmentation and segmentation != []:
                        annotation["segmentation"] = segmentation
                    
                    annotations.append(annotation)
                    cur_ann_ids.append(annotation_id_counter)
                    annotation_id_counter += 1  # ← Increment
                    # cur_ann_ids.append(frame_obj_idx)  # Not annotation["id"]!
                    # frame_obj_idx += 1

                    # cur_ann_ids.append(annotation["id"])
                
                # ✅ CRITICAL DEBUG: Check before creating query
                if len(cur_ann_ids) == 0:
                    print(f"\n  ❌❌❌ WARNING: Creating query with EMPTY annotations!")
                    print(f"    Frame {frame_idx}, Category {cat_id}")
                    print(f"    Annotations attempted: {annotations_attempted}")
                    print(f"    Annotations skipped: {annotations_skipped}")
                    print(f"    frame_cat_has_data says this should have data!")
                    
                    # # Debug why all were skipped
                    # for obj_ann in anns:
                    #     print(f"    Ann {obj_ann['id']}:")
                    #     if 'frame_indices' in obj_ann:
                    #         has_frame = frame_idx in obj_ann['frame_indices']
                    #         print(f"      Has frame {frame_idx}: {has_frame}")
                    #         if has_frame:
                    #             idx = obj_ann['frame_indices'].index(frame_idx)
                    #             print(f"      Bbox: {obj_ann['bboxes'][idx]}")
                    #     elif 'bboxes' in obj_ann:
                    #         if frame_idx < len(obj_ann['bboxes']):
                    #             print(f"      Bbox: {obj_ann['bboxes'][frame_idx]}")
                    
                    queries_with_empty_anns += 1
                    # ❌ DON'T CREATE THIS QUERY
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
                queries_created += 1
    
        
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
        # === gaoqi:【新增】处理 is_exhaustive 字段 ===
        # 优先从 JSON 文件的 image 信息中读取 is_instance_exhaustive
        # 如果没有，则默认设置为 False（弱监督 scribble 场景）
        if "is_instance_exhaustive" in cur_img_data:
            query["is_exhaustive"] = bool(cur_img_data["is_instance_exhaustive"])
        else:
            # 对于弱监督 scribble 标注，默认设置为 False
            query["is_exhaustive"] = False
        # ==========================================
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
