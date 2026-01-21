# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# 针对弱监督任务进行修改
"""Dataset class for modulated detection"""

import json
import os
import random
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.data
import torchvision
from decord import cpu, VideoReader
from iopath.common.file_io import g_pathmgr

from PIL import Image as PILImage
from PIL.Image import DecompressionBombError

from sam3.model.box_ops import box_xywh_to_xyxy
from torchvision.datasets.vision import VisionDataset

from .coco_json_loaders import COCO_FROM_JSON

# 在 import 部分添加
from pycocotools import mask as maskUtils
import numpy as np


@dataclass
class InferenceMetadata:
    """Metadata required for postprocessing"""

    # Coco id that corresponds to the "image" for evaluation by the coco evaluator
    # This is used for our own "class agnostic" evaluation
    coco_image_id: int

    # id in the original dataset, such that we can use the original evaluator
    original_image_id: int

    # Original category id (if we want to use the original evaluator)
    original_category_id: int

    # Size of the raw image (height, width)
    original_size: Tuple[int, int]

    # Id of the object in the media
    object_id: int

    # Index of the frame in the media (0 if single image)
    frame_index: int

    # Whether it is for conditioning only, e.g., 0-th frame in TA is for conditioning
    # as we assume GT available in frame 0.
    is_conditioning_only: Optional[bool] = False


@dataclass
class FindQuery:
    query_text: str

    image_id: int

    # In case of a find query, the list of object ids that have to be predicted
    object_ids_output: List[int]

    # This is "instance exhaustivity".
    # true iff all instances are separable and annotated
    # See below the slightly different "pixel exhaustivity"
    is_exhaustive: bool

    # The order in which the queries are processed (only meaningful for video)
    query_processing_order: int = 0

    # Input geometry, initially in denormalized XYXY format. Then
    # 1. converted to normalized CxCyWH by the Normalize transform
    input_bbox: Optional[torch.Tensor] = None
    input_bbox_label: Optional[torch.Tensor] = None

    # Only for the PVS task
    input_points: Optional[torch.Tensor] = None

    semantic_target: Optional[torch.Tensor] = None

    # pixel exhaustivity: true iff the union of all segments (including crowds)
    # covers every pixel belonging to the target class
    # Note that instance_exhaustive implies pixel_exhaustive
    is_pixel_exhaustive: Optional[bool] = None


@dataclass
class FindQueryLoaded(FindQuery):
    # Must have default value since FindQuery has entries with default values
    inference_metadata: Optional[InferenceMetadata] = None


@dataclass
class Object:
    # Initially in denormalized XYXY format, gets converted to normalized CxCyWH by the Normalize transform
    bbox: torch.Tensor
    area: float

    # Id of the object in the media
    object_id: Optional[int] = -1

    # Index of the frame in the media (0 if single image)
    frame_index: Optional[int] = -1

    segment: Optional[Union[torch.Tensor, dict]] = None  # RLE dict or binary mask

    is_crowd: bool = False

    source: Optional[str] = None


@dataclass
class Image:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    size: Tuple[int, int]  # (height, width)

    # For blurring augmentation
    blurring_mask: Optional[Dict[str, Any]] = None


@dataclass
class Datapoint:
    """Refers to an image/video and all its annotations"""

    find_queries: List[FindQueryLoaded]
    images: List[Image]
    raw_images: Optional[List[PILImage.Image]] = None


class CustomCocoDetectionAPI(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        load_segmentation: bool,
        fix_fname: bool = False,
        training: bool = True,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        limit_ids: int = None,
    ) -> None:
        super().__init__(root)

        self.annFile = annFile
        self.use_caching = use_caching
        self.zstd_dict_path = zstd_dict_path

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.load_segmentation = load_segmentation
        self.fix_fname = fix_fname
        self.filter_query = filter_query

        self.coco = None
        self.coco_json_loader = coco_json_loader
        self.limit_ids = limit_ids
        self.set_sharded_annotation_file(0)
        self.training = training
        self.blurring_masks_path = blurring_masks_path

    def _load_images(
        self, datapoint_id: int, img_ids_to_load: Optional[Set[int]] = None
    ) -> Tuple[List[Tuple[int, PILImage.Image]], List[Dict[str, Any]]]:
        all_images = []
        all_img_metadata = []
        for current_meta in self.coco.loadImagesFromDatapoint(datapoint_id):
            img_id = current_meta["id"]
            if img_ids_to_load is not None and img_id not in img_ids_to_load:
                continue
            if self.fix_fname:
                current_meta["file_name"] = current_meta["file_name"].split("/")[-1]
            path = current_meta["file_name"]
            if self.blurring_masks_path is not None:
                mask_fname = os.path.basename(path).replace(".jpg", "-mask.json")
                mask_path = os.path.join(self.blurring_masks_path, mask_fname)
                if os.path.exists(mask_path):
                    with open(mask_path, "r") as fopen:
                        current_meta["blurring_mask"] = json.load(fopen)

            all_img_metadata.append(current_meta)
            path = os.path.join(self.root, path)
            try:
                if ".mp4" in path and path[-4:] == ".mp4":
                    # Going to load a video frame
                    video_path, frame = path.split("@")
                    video = VideoReader(video_path, ctx=cpu(0))
                    # Convert to PIL image
                    all_images.append(
                        (
                            img_id,
                            torchvision.transforms.ToPILImage()(
                                video[int(frame)].asnumpy()
                            ),
                        )
                    )
                else:
                    with g_pathmgr.open(path, "rb") as fopen:
                        all_images.append((img_id, PILImage.open(fopen).convert("RGB")))
            except FileNotFoundError as e:
                print(f"File not found: {path} from dataset: {self.annFile}")
                raise e

        return all_images, all_img_metadata

    def set_curr_epoch(self, epoch: int):
        self.curr_epoch = epoch

    def set_epoch(self, epoch: int):
        pass

    def set_sharded_annotation_file(self, data_epoch: int):
        if self.coco is not None:
            return

        assert g_pathmgr.isfile(
            self.annFile
        ), f"please provide valid annotation file. Missing: {self.annFile}"
        annFile = g_pathmgr.get_local_path(self.annFile)

        if self.coco is not None:
            del self.coco

        self.coco = self.coco_json_loader(annFile)
        # Use a torch tensor here to optimize memory usage when using several dataloaders
        ids_list = list(sorted(self.coco.getDatapointIds()))
        if self.limit_ids is not None:
            local_random = random.Random(len(ids_list))
            local_random.shuffle(ids_list)
            ids_list = ids_list[: self.limit_ids]
        self.ids = torch.as_tensor(ids_list, dtype=torch.long)

    def __getitem__(self, index: int) -> Datapoint:
        return self._load_datapoint(index)

    def _load_datapoint(self, index: int) -> Datapoint:
        """A separate method for easy overriding in subclasses."""
        id = self.ids[index].item()
        pil_images, img_metadata = self._load_images(id)
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)
        return self.load_queries(pil_images, annotations, queries, img_metadata)

    def load_queries(self, pil_images, annotations, queries, img_metadata):
        """Transform the raw image and queries into a Datapoint sample."""
        images: List[Image] = []
        id2index_img = {}
        id2index_obj = {}
        id2index_find_query = {}
        id2imsize = {}
        assert len(pil_images) == len(img_metadata)
        for i in range(len(pil_images)):
            w, h = pil_images[i][1].size
            blurring_mask = None
            if "blurring_mask" in img_metadata[i]:
                blurring_mask = img_metadata[i]["blurring_mask"]
            images.append(
                Image(
                    data=pil_images[i][1],
                    objects=[],
                    size=(h, w),
                    blurring_mask=blurring_mask,
                )
            )
            id2index_img[pil_images[i][0]] = i
            id2imsize[pil_images[i][0]] = (h, w)

        # === gaoqi:优化重点：Annotations 循环 ===
        for annotation in annotations:
            image_id = id2index_img[annotation["image_id"]]
            h, w = id2imsize[annotation["image_id"]]

            # -----------------------------------------------------------
            # 步骤 1: 预先解码当前对象的 Mask (只解码一次！)
            # -----------------------------------------------------------
            current_binary_mask = None
            if isinstance(annotation["segmentation"], list):
                # Polygon 格式
                rle = maskUtils.frPyObjects(annotation["segmentation"], h, w)
                m = maskUtils.decode(rle)
                if len(m.shape) == 3:
                    m = m.sum(axis=2) > 0
                current_binary_mask = m.astype(bool)
            elif isinstance(annotation["segmentation"], dict):
                # RLE 格式
                m = maskUtils.decode(annotation["segmentation"])
                current_binary_mask = m.astype(bool)
            else:
                # 异常处理
                current_binary_mask = np.zeros((h, w), dtype=bool)

            # -----------------------------------------------------------
            # 步骤 2: 获取 BBox ((如果 JSON 中没有，则从 Mask 生成伪 Box))
            # -----------------------------------------------------------
            # 优先级：推理得到的bbox > JSON中的bbox > 从scribble mask计算的bbox
            ann_id = annotation.get("id", -1)
            use_inferred_bbox = False
            
            # 检查是否有推理得到的bbox
            if hasattr(self, 'inferred_bbox_cache') and ann_id in self.inferred_bbox_cache:
                inferred_bbox_xywh = self.inferred_bbox_cache[ann_id]
                if inferred_bbox_xywh is not None and len(inferred_bbox_xywh) == 4:
                    # 使用推理得到的bbox
                    bbox_xywh = torch.as_tensor(inferred_bbox_xywh)
                    bbox = box_xywh_to_xyxy(bbox_xywh).view(1, 4)
                    bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
                    bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
                    use_inferred_bbox = True
            
            if not use_inferred_bbox:
                # COCO Loader 已经读取了 "bbox" 并进行了归一化 (0-1)
                # 检查 JSON 里的 bbox 是否有效 (宽和高 > 0)
                has_valid_json_bbox = False
                if "bbox" in annotation:
                    temp_bbox = torch.as_tensor(annotation["bbox"])
                    # 注意：coco_json_loaders 可能会填充 [0,0,0,0]，需要过滤掉
                    if temp_bbox[2] > 0 and temp_bbox[3] > 0:
                        has_valid_json_bbox = True
                
                if has_valid_json_bbox:
                    # 原始逻辑：从 JSON 加载并反归一化
                    bbox_xywh = torch.as_tensor(annotation["bbox"])
                    bbox = box_xywh_to_xyxy(bbox_xywh).view(1, 4)
                    bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
                    bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
                else:
                    # === 新增逻辑：从 Scribble Mask 计算 Pseudo BBox ===
                    if current_binary_mask.any():
                        y_indices, x_indices = np.where(current_binary_mask)
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        # 构造绝对坐标 XYXY
                        # 稍微给一点 buffer (可选)，或者直接紧贴 scribble
                        bbox = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
                    else:
                        # 空 Mask 的情况 (防御性代码)
                        bbox = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

            # -----------------------------------------------------------
            # 步骤 3: 生成 Mask (全监督: 0/1, 弱监督: 0/1/255)
            # -----------------------------------------------------------
            segment = None
            if self.load_segmentation: # 只要开启分割就生成
                # 检测是否为弱监督任务（通过 valid_mask 判断）
                has_valid_mask = "valid_mask" in annotation and annotation["valid_mask"] is not None
                
                if has_valid_mask:
                    # === 弱监督任务：生成 Tri-state Mask (0=负, 1=正, 255=忽略) ===
                    # 1. 初始化 Ignore (255)
                    tri_state_mask = np.full((h, w), 255, dtype=np.uint8)

                    # 2. 解码 valid_mask 并设定背景 (0)
                    # valid_mask=1 的区域是所有已知 scribble 的并集
                    rle_valid = annotation["valid_mask"]
                    if isinstance(rle_valid, list): # 兼容 polygon
                        rle_valid = maskUtils.frPyObjects(rle_valid, h, w)
                    m_valid = maskUtils.decode(rle_valid)
                    if len(m_valid.shape) == 3: m_valid = m_valid.sum(axis=2) > 0
                    
                    # 将所有有效区域设为 0 (背景)
                    # 稍后会将正样本覆盖为 1
                    tri_state_mask[m_valid.astype(bool)] = 0
                    
                    # 3. 设定正样本 (1) - 使用 Step 1 中已经解码好的 current_binary_mask
                    if current_binary_mask is not None:
                        tri_state_mask[current_binary_mask] = 1
                    
                    # 4. 转 Tensor
                    segment = torch.from_numpy(tri_state_mask).long()
                else:
                    # === 全监督任务：生成二值 Mask (0=背景, 1=前景) ===
                    if current_binary_mask is not None:
                        # 直接使用二值 mask：True -> 1, False -> 0
                        binary_mask = current_binary_mask.astype(np.uint8)
                    else:
                        # 防御性代码：如果没有 mask，创建全零 mask
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 转 Tensor
                    segment = torch.from_numpy(binary_mask).long()

            images[image_id].objects.append(
                Object(
                    bbox=bbox[0],
                    area=annotation["area"],
                    object_id=(
                        annotation["object_id"] if "object_id" in annotation else -1
                    ),
                    frame_index=(
                        annotation["frame_index"] if "frame_index" in annotation else -1
                    ),
                    segment=segment, # 这里现在是 [H, W] 的 0/1/255 Tensor
                    is_crowd=(
                        annotation["is_crowd"] if "is_crowd" in annotation else None
                    ),
                    source=annotation["source"] if "source" in annotation else "",
                )
            )
            id2index_obj[annotation["id"]] = len(images[image_id].objects) - 1

        find_queries = []
        stage2num_queries = Counter()
        for i, query in enumerate(queries):
            stage2num_queries[query["query_processing_order"]] += 1
            id2index_find_query[query["id"]] = i

        # Sanity check: all the stages should have the same number of queries
        if len(stage2num_queries) == 0:
            num_queries_per_stage = 0
        else:
            num_queries_per_stage = stage2num_queries.most_common(1)[0][1]
        for stage, num_queries in stage2num_queries.items():
            assert (
                num_queries == num_queries_per_stage
            ), f"Number of queries in stage {stage} is {num_queries}, expected {num_queries_per_stage}"

        for query_id, query in enumerate(queries):
            h, w = id2imsize[query["image_id"]]
            if (
                "input_box" in query
                and query["input_box"] is not None
                and len(query["input_box"]) > 0
            ):
                bbox = box_xywh_to_xyxy(torch.as_tensor(query["input_box"])).view(-1, 4)
                bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
                bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
                if "input_box_label" in query and query["input_box_label"] is not None:
                    bbox_label = torch.as_tensor(
                        query["input_box_label"], dtype=torch.long
                    ).view(-1)
                    assert len(bbox_label) == len(bbox)
                else:
                    # assume the boxes are positives
                    bbox_label = torch.ones(len(bbox), dtype=torch.long)
            else:
                bbox = None
                bbox_label = None

            if "input_points" in query and query["input_points"] is not None:
                points = torch.as_tensor(query["input_points"]).view(1, -1, 3)
                points[:, :, 0:1].mul_(w).clamp_(min=0, max=w)
                points[:, :, 1:2].mul_(h).clamp_(min=0, max=h)
            else:
                points = None

            try:
                original_image_id = int(
                    img_metadata[id2index_img[query["image_id"]]]["original_img_id"]
                )
            except ValueError:
                original_image_id = -1

            try:
                img_metadata_query = img_metadata[id2index_img[query["image_id"]]]
                coco_image_id = (
                    int(img_metadata_query["coco_img_id"])
                    if "coco_img_id" in img_metadata_query
                    else query["id"]
                )
            except KeyError:
                coco_image_id = -1

            try:
                original_category_id = int(query["original_cat_id"])
            except (ValueError, KeyError):
                original_category_id = -1

            # For evaluation, we associate the ids of the object to be tracked to the query
            if query["object_ids_output"]:
                obj_id = query["object_ids_output"][0]
                obj_idx = id2index_obj[obj_id]
                image_idx = id2index_img[query["image_id"]]
                object_id = images[image_idx].objects[obj_idx].object_id
                frame_index = images[image_idx].objects[obj_idx].frame_index
            else:
                object_id = -1
                frame_index = -1

            find_queries.append(
                FindQueryLoaded(
                    # id=query["id"],
                    # query_type=qtype,
                    query_text=(
                        query["query_text"] if query["query_text"] is not None else ""
                    ),
                    image_id=id2index_img[query["image_id"]],
                    input_bbox=bbox,
                    input_bbox_label=bbox_label,
                    input_points=points,
                    object_ids_output=[
                        id2index_obj[obj_id] for obj_id in query["object_ids_output"]
                    ],
                    is_exhaustive=query["is_exhaustive"],
                    is_pixel_exhaustive=(
                        query["is_pixel_exhaustive"]
                        if "is_pixel_exhaustive" in query
                        else (
                            query["is_exhaustive"] if query["is_exhaustive"] else None
                        )
                    ),
                    query_processing_order=query["query_processing_order"],
                    inference_metadata=InferenceMetadata(
                        coco_image_id=-1 if self.training else coco_image_id,
                        original_image_id=(-1 if self.training else original_image_id),
                        frame_index=frame_index,
                        original_category_id=original_category_id,
                        original_size=(h, w),
                        object_id=object_id,
                    ),
                )
            )

        return Datapoint(
            find_queries=find_queries,
            images=images,
            raw_images=[p[1] for p in pil_images],
        )

    def __len__(self) -> int:
        return len(self.ids)


class Sam3ImageDataset(CustomCocoDetectionAPI):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        max_ann_per_img: int,
        multiplier: int,
        training: bool,
        load_segmentation: bool = False,
        max_train_queries: int = 81,
        max_val_queries: int = 300,
        fix_fname: bool = False,
        is_sharded_annotation_dir: bool = False,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        limit_ids: int = None,
    ):
        super(Sam3ImageDataset, self).__init__(
            img_folder,
            ann_file,
            fix_fname=fix_fname,
            load_segmentation=load_segmentation,
            training=training,
            blurring_masks_path=blurring_masks_path,
            use_caching=use_caching,
            zstd_dict_path=zstd_dict_path,
            filter_query=filter_query,
            coco_json_loader=coco_json_loader,
            limit_ids=limit_ids,
        )

        self._transforms = transforms
        self.training = training
        self.max_ann_per_img = max_ann_per_img
        self.max_train_queries = max_train_queries
        self.max_val_queries = max_val_queries

        self.repeat_factors = torch.ones(len(self.ids), dtype=torch.float32)

        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.ids)}")

        self._MAX_RETRIES = 100
        
        # 用于存储推理得到的bbox缓存: {annotation_id: bbox_xywh}
        # bbox_xywh是归一化的格式 [x, y, w, h]
        self.inferred_bbox_cache: Dict[int, List[float]] = {}

    def __getitem__(self, idx):
        return self.__orig_getitem__(idx)

    def __orig_getitem__(self, idx):
        for _ in range(self._MAX_RETRIES):
            try:
                datapoint = super(Sam3ImageDataset, self).__getitem__(idx)

                # This can be done better by filtering the offending find queries
                # However, this requires care:
                # - Delete any find/get query that may depend on the deleted one
                # - Re-compute the indexes in the pointers to account for the deleted finds
                for q in datapoint.find_queries:
                    if len(q.object_ids_output) > self.max_ann_per_img:
                        raise DecompressionBombError(
                            f"Too many outputs ({len(q.object_ids_output)})"
                        )

                max_queries = (
                    self.max_train_queries if self.training else self.max_val_queries
                )

                if len(datapoint.find_queries) > max_queries:
                    raise DecompressionBombError(
                        f"Too many find queries ({len(datapoint.find_queries)})"
                    )

                if len(datapoint.find_queries) == 0:
                    raise DecompressionBombError("No find queries")
                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)

                break
            except (DecompressionBombError, OSError, ValueError) as error:
                sys.stderr.write(f"ERROR: got loading error on datapoint {idx}\n")
                sys.stderr.write(f"Exception: {error}\n")
                sys.stderr.write(traceback.format_exc())
                idx = (idx + 1) % len(self)
        else:
            raise RuntimeError(
                f"Failed {self._MAX_RETRIES} times trying to load an image."
            )

        return datapoint
    
    def update_inferred_bbox_cache(self, bbox_cache: Dict[int, List[float]]):
        """
        更新推理得到的bbox缓存
        
        Args:
            bbox_cache: Dict[annotation_id, bbox_xywh]
                bbox_xywh是归一化的格式 [x, y, w, h]
        """
        import torch.distributed as dist
        
        self.inferred_bbox_cache.update(bbox_cache)
        
        # 获取当前rank（如果是分布式训练）
        # 使用print而不是logging，因为nohup会捕获所有rank的stdout
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"Rank {rank}: 更新了 {len(bbox_cache)} 个推理得到的bbox")
    
    def clear_inferred_bbox_cache(self):
        """清空推理得到的bbox缓存"""
        import torch.distributed as dist
        
        self.inferred_bbox_cache.clear()
        
        # 获取当前rank（如果是分布式训练）
        # 使用print而不是logging，因为nohup会捕获所有rank的stdout
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"Rank {rank}: 已清空推理得到的bbox缓存")