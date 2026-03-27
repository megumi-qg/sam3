# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import copy
import numpy as np
from PIL import Image
import os
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torchvision
from .sam3_image_dataset import Datapoint, Sam3ImageDataset
from .coco_json_loaders import COCO_VIDEO_FROM_JSON

SEED = 42


class VideoGroundingDataset(Sam3ImageDataset):
    def __init__(
        self,
        num_stages_sample: int = 4,
        stage_stride_min: int = 1,
        stage_stride_max: int = 5,
        random_reverse_time_axis: bool = True,
        is_tiling_single_image: bool = False,
        # By default, we remove find those queries with geometric inputs (input_box or input_points)
        # when creating synthetic videos from frames (since they are not *video-level* text prompts).
        # If we need them later, we can sample them on-the-fly via transforms or inside the model.
        tile_img_keep_find_queries_with_geo_inputs: bool = False,
        tile_img_keep_get_queries: bool = False,
        # the maximum number of find queries (for each frame) to keep in a video; if the datapoint
        # contains more queries per frame than this limit, we subsample them to avoid OOM errors
        max_query_num: int = -1,  # the default -1 means no limit
        # whether to override the "is_exhaustive" flag of the loaded find queries to True
        # (by default, our video datasets are ingested with is_exhaustive=False, since the YTVIS format
        # annotations doesn't involve an "is_exhaustive" flag; this means that those unmatched (negative)
        # detection queries or tracking queries do not receive a classification loss given that we have
        # weak_loss=True in IABCEMdetr -- this could lead to false positives for both image detection
        # and video association.)
        override_query_is_exhaustive_to_true: bool = False,
        # the maximum number of masklets in a video; if the datapoint contains more masklets
        # than this limit, we skip the datapoint to avoid OOM errors (this is useful for
        # training with large videos that contain many objects)
        max_masklet_num_in_video: int = 300,  # 300 masklets is usually OK to avoid OOM
        **kwargs,
    ):
        """
        Loading video grounding data

        Video frame sampling parameters (for training only):
        - num_stages_sample: number of frames to sample from the video during training
        - stage_stride_min: minimum stride between sampled frames during training
        - stage_stride_max: maximum stride between sampled frames during training (if it's
          greater than stage_stride_min, the actual stride is sampled uniformly between min
          and max; during inference, we always use all frames in the video with stride=1)
        - random_reverse_time_axis: whether to randomly invert the video's temporal axis
          (i.e. playing it backwards) during training
        """
        # ✅ 修改点 1：显式传递 coco_json_loader 为你的自定义类
        # 注意：这里我们通过 kwargs 将其传递给 Sam3ImageDataset -> CustomCocoDetectionAPI
        if "coco_json_loader" not in kwargs:
            kwargs["coco_json_loader"] = COCO_VIDEO_FROM_JSON
        
        super().__init__(**kwargs)
        assert num_stages_sample >= 1
        assert stage_stride_min >= 1
        assert stage_stride_max >= stage_stride_min
        self.num_stages_sample = num_stages_sample
        self.stage_stride_min = stage_stride_min
        self.stage_stride_max = stage_stride_max
        self.random_reverse_time_axis = random_reverse_time_axis
        self.is_tiling_single_image = is_tiling_single_image
        self.tile_img_keep_find_queries_with_geo_inputs = (
            tile_img_keep_find_queries_with_geo_inputs
        )
        self.tile_img_keep_get_queries = tile_img_keep_get_queries
        self.max_query_num = max_query_num
        self.override_query_is_exhaustive_to_true = override_query_is_exhaustive_to_true
        self.max_masklet_num_in_video = max_masklet_num_in_video
        self.rng = random.Random()
        self.set_curr_epoch(0)
    
    # ✅ 修改点 2：重写 _load_images 以支持 NPZ 切片加载
    def _load_images(
        self, datapoint_id: int, img_ids_to_load: Optional[Set[int]] = None
    ) -> Tuple[List[Tuple[int, Image.Image]], List[Dict[str, Any]]]:
        
        all_images = []
        all_img_metadata = []
        
        # 使用你的 COCO_VIDEO_FROM_JSON 类获取元数据
        images_meta = self.coco.loadImagesFromDatapoint(datapoint_id)
        
        for current_meta in images_meta:
            img_id = current_meta["id"]
            
            # 过滤不需要的帧（根据时间采样逻辑）
            if img_ids_to_load is not None and img_id not in img_ids_to_load:
                continue
            
            # 处理文件名（兼容基类逻辑）
            if self.fix_fname:
                current_meta["file_name"] = current_meta["file_name"].split("/")[-1]
            
            # 记录元数据
            all_img_metadata.append(current_meta)
            
            # 获取文件绝对路径
            path = os.path.join(self.root, current_meta["file_name"])

            # ----------------------------------------------------------------
            #  新增逻辑：处理 NPZ 格式的医学图像切片
            # ----------------------------------------------------------------
            if current_meta.get("is_npz", False):
                try:
                    slice_idx = current_meta["frame_idx"]
                    
                    # 读取 NPZ
                    # ⚠️ 性能提示：如果在同一个 batch 中频繁读取同一个大的 npz 文件，
                    # 建议在这里增加一个 LRU Cache 或者将 data 加载到内存中，
                    # 否则频繁 IO 会导致训练速度变慢。
                    with np.load(path) as data:
                        # 假设数据存储在第一个 key 下，或者是 'arr_0'，或者是 'volume'
                        # 根据你的 npz 结构调整 key
                        keys = list(data.keys())
                        vol_key = 'volume' if 'volume' in keys else keys[0]
                        volume = data[vol_key]
                        
                        # 提取特定层
                        # 假设 volume 是 [D, H, W]
                        if slice_idx < volume.shape[0]:
                            img_array = volume[slice_idx]
                        else:
                            raise ValueError(f"Slice index {slice_idx} out of bounds for {path}")
                    
                    # 归一化并转为 PIL (SAM3 需要 RGB PIL 输入)
                    img_pil = self._numpy_to_pil(img_array)
                    all_images.append((img_id, img_pil))
                    
                except Exception as e:
                    print(f"Error loading NPZ slice: {path}, index: {slice_idx}")
                    raise e
            
            # ----------------------------------------------------------------
            #  原有逻辑：处理常规视频或图片 (保留以兼容混合数据集)
            # ----------------------------------------------------------------
            else:
                try:
                    # 处理 MP4
                    if ".mp4" in path and path.endswith(".mp4"):
                        # 这里沿用父类逻辑，如果父类引用不可用，需要复制父类相关代码
                        # 假设 path 格式为 "video.mp4@frame_idx" 如果不是，需要适配
                        if "@" in path:
                            video_path, frame = path.split("@")
                            # 需要确保 VideoReader 可用
                            from decord import VideoReader, cpu
                            video = VideoReader(video_path, ctx=cpu(0))
                            pil_img = torchvision.transforms.ToPILImage()(
                                video[int(frame)].asnumpy()
                            )
                            all_images.append((img_id, pil_img))
                        else: 
                            # 如果只是普通 MP4 文件名，需要根据 frame_idx 读取
                            frame_idx = current_meta.get("frame_idx", 0)
                            from decord import VideoReader, cpu
                            video = VideoReader(path, ctx=cpu(0))
                            pil_img = torchvision.transforms.ToPILImage()(
                                video[frame_idx].asnumpy()
                            )
                            all_images.append((img_id, pil_img))

                    # 处理普通图片 (JPG/PNG)
                    else:
                        with open(path, "rb") as fopen:
                            img = Image.open(fopen).convert("RGB")
                            all_images.append((img_id, img))
                            
                except FileNotFoundError as e:
                    print(f"File not found: {path} from dataset: {self.annFile}")
                    raise e

        return all_images, all_img_metadata

    def _numpy_to_pil(self, img_array):
        """
        辅助函数：将任意范围的 numpy 数组 (e.g. CT 值) 转换为 RGB PIL Image
        """
        # 1. 归一化到 0-255
        if img_array.max() > img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        else:
            img_array = np.zeros_like(img_array)
            
        img_array = (img_array * 255).astype(np.uint8)
        
        # 2. 转为 PIL
        img_pil = Image.fromarray(img_array)
        
        # 3. 如果是单通道，转为 RGB (SAM3 需要)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
            
        return img_pil


    def set_curr_epoch(self, epoch: int):
        super().set_curr_epoch(epoch)
        self.rng.seed(SEED + epoch)

    def _load_datapoint(self, index: int) -> Datapoint:
        id = self.ids[index].item()
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)

        # we subsample the video frames during training
        if self.training and not self.is_tiling_single_image:
            # pick a random stride for sampling query stages (`randint` includes both ends)
            stage_stride = self.rng.randint(
                self.stage_stride_min, self.stage_stride_max
            )
            stage_ids_to_keep = self._sample_stage_ids(
                queries, self.num_stages_sample, stage_stride
            )
            # filter the queries and annotations to keep only the selected stages
            # (also remap the stage ids so that they are contiguous and start from 0)
            reverse_time_axis = (
                self.rng.random() < 0.5 if self.random_reverse_time_axis else False
            )
            queries, annotations, kept_img_ids = self._filter_query_and_anns(
                queries,
                annotations,
                stage_ids_to_keep,
                remap_stage_id=True,
                reverse_time_axis=reverse_time_axis,
            )
            pil_images, img_metadata = self._load_images(id, kept_img_ids)
            if reverse_time_axis:
                # reverse the temporal ordering of the images and their metadata
                # so that the image order matches the query order
                pil_images = pil_images[::-1]
                img_metadata = img_metadata[::-1]
        else:
            pil_images, img_metadata = self._load_images(id)

        # check that all the images have the same image size (they are expected
        # to have the same image size since they are frames from the same video)
        assert all(p.size == pil_images[0][1].size for _, p in pil_images)

        queries.sort(key=lambda q: q["query_processing_order"])
        if self.override_query_is_exhaustive_to_true:
            for query in queries:
                query["is_exhaustive"] = True
        datapoint = self.load_queries(pil_images, annotations, queries, img_metadata)

        # skip datapoints with too many masklets to avoid OOM errors
        num_masklets_in_video = len(datapoint.images[0].objects)
        if num_masklets_in_video > self.max_masklet_num_in_video > 0:
            logging.warning(
                f"Datapoint {id} has ({num_masklets_in_video=}), exceeding "
                f"the maximum allowed ({self.max_masklet_num_in_video}). "
                "Skipping this datapoint."
            )
            next_index = (index + 1) % len(self)
            return self._load_datapoint(next_index)  # move to the next datapoint

        if self.is_tiling_single_image:
            datapoint = self._tile_single_image_data(datapoint, self.num_stages_sample)
        if self.max_query_num > 0:
            datapoint = self._subsample_queries(datapoint, self.max_query_num)

        # ensure that all find queries have the same processing order as their image id
        for query in datapoint.find_queries:
            assert query.image_id == query.query_processing_order, (
                f"find query has inconsistent image_id and "
                f"query_processing_order: {query.image_id=} vs "
                f"{query.query_processing_order=}"
            )
        return datapoint

    def _sample_stage_ids(self, queries, num_stages_sample, stage_stride):
        """Sample a subset of stage ids from all queries."""
        # Later we can perhaps turn it into a Sampler class to be more flexible.
        all_stage_ids = sorted(set(q["query_processing_order"] for q in queries))
        num_stages_total = len(all_stage_ids)
        if num_stages_total < num_stages_sample:
            raise ValueError("Not enough stages to sample")

        # the difference in index between the first and the last sampled stage ids
        b_e_gap = (num_stages_sample - 1) * stage_stride
        if b_e_gap > num_stages_total - 1:
            # In this case, it's not possible to sample with the provide stride,
            # so we use the maximum possible stride.
            prev_stage_stride = stage_stride
            stage_stride = math.floor((num_stages_total - 1) / (num_stages_sample - 1))
            logging.info(
                f"lowering stride from {prev_stage_stride} to {stage_stride} to "
                f"sample {num_stages_sample} stages (from {num_stages_total} total)"
            )
            b_e_gap = (num_stages_sample - 1) * stage_stride

        # randomly select a starting stage id (`randint` includes both ends)
        b_max = len(all_stage_ids) - 1 - b_e_gap
        b = self.rng.randint(0, b_max)
        e = b + b_e_gap
        stage_ids_to_keep = all_stage_ids[b : e + 1 : stage_stride]
        return stage_ids_to_keep

    def _filter_query_and_anns(
        self, queries, annotations, stage_ids_to_keep, remap_stage_id, reverse_time_axis
    ):
        """Filter queries and annotations to only keep those in `stage_ids_to_keep`."""
        stage_ids_to_keep = set(stage_ids_to_keep)
        kept_img_ids = set()
        kept_stage_ids = set()

        # Filter queries -- keep those queries with stage_id in `stage_ids_to_keep`
        filtered_queries = []
        for query in queries:
            input_box = query.get("input_box", None)
            input_points = query.get("input_points", None)
            has_geo_input = input_box is not None or input_points is not None
            if has_geo_input and not self.tile_img_keep_find_queries_with_geo_inputs:
                continue
            stage_id = query["query_processing_order"]
            if stage_id in stage_ids_to_keep:
                kept_img_ids.add(query["image_id"])
                kept_stage_ids.add(stage_id)
                filtered_queries.append(query)
        # Check that all frames in `stage_ids_to_keep` are present after filtering
        all_frame_present = kept_stage_ids == stage_ids_to_keep
        assert all_frame_present, f"{kept_stage_ids=} vs {stage_ids_to_keep=}"
        if remap_stage_id:
            # Remap those kept stage ids to be contiguous and starting from 0
            old_stage_ids = sorted(kept_stage_ids, reverse=reverse_time_axis)
            stage_id_old2new = {old: new for new, old in enumerate(old_stage_ids)}
            for query in filtered_queries:
                ptr_x_is_empty = query["ptr_x_query_id"] in [None, -1]
                ptr_y_is_empty = query["ptr_y_query_id"] in [None, -1]
                assert (
                    ptr_x_is_empty and ptr_y_is_empty
                ), "Remapping stage ids is not supported for queries with non-empty ptr_x or ptr_y pointers"
                query["query_processing_order"] = stage_id_old2new[
                    query["query_processing_order"]
                ]

        # Filter annotations -- keep those annotations with image_id in `kept_img_ids`
        filtered_annotations = [
            ann for ann in annotations if ann["image_id"] in kept_img_ids
        ]

        return filtered_queries, filtered_annotations, kept_img_ids

    def _tile_single_image_data(self, datapoint: Datapoint, num_stages_sample: int):
        """
        Tile a single image and its queries to simulate video frames. The output is a
        datapoint with *identical video frames* (i.e. the same static image) and needs
        further transforms (e.g. affine) to get video frames with different content.
        """
        # tile `images: List[Image]`
        assert len(datapoint.images) == 1, "Expected only one single image"
        tiled_images = [
            copy.deepcopy(datapoint.images[0]) for _ in range(num_stages_sample)
        ]
        for stage_id, img in enumerate(tiled_images):
            for obj in img.objects:
                obj.frame_index = stage_id

        # tile `raw_images: Optional[List[PILImage.Image]] = None`
        tiled_raw_images = None
        if datapoint.raw_images is not None:
            assert len(datapoint.raw_images) == 1, "Expected only one single image"
            tiled_raw_images = [
                datapoint.raw_images[0].copy() for _ in range(num_stages_sample)
            ]

        # tile `find_queries: List[FindQueryLoaded]`
        tiled_find_queries_per_stage = [[] for _ in range(num_stages_sample)]
        for query in datapoint.find_queries:
            assert query.image_id == 0
            assert query.query_processing_order == 0
            # check and make sure that a query doesn't contain pointers or references
            # to other queries (that cannot be tiled)
            assert query.ptr_x is None and query.ptr_y is None
            assert query.ptr_mem is None
            # assert query.wkdata_qid is None
            # assert query.other_positive_qids is None
            # assert query.negative_qids is None
            has_geo_input = (
                query.input_bbox is not None or query.input_points is not None
            )
            if has_geo_input and not self.tile_img_keep_find_queries_with_geo_inputs:
                continue
            for stage_id in range(num_stages_sample):
                # copy the query and update the image_id
                new_query = copy.deepcopy(query)
                new_query.image_id = stage_id
                new_query.query_processing_order = stage_id
                if new_query.inference_metadata is not None:
                    new_query.inference_metadata.frame_index = stage_id
                tiled_find_queries_per_stage[stage_id].append(new_query)

        tiled_find_queries = sum(tiled_find_queries_per_stage, [])

        # tile `get_queries: List[GetQuery]` -- we skip them for now (since they involve
        # a pointer to a find query that is complicated to tile, and there is not an
        # imminent use case for them in the video grounding task in the near future)
        if self.tile_img_keep_get_queries:
            raise NotImplementedError("Tiling get queries is not implemented yet")
        else:
            tiled_get_queries = []

        return Datapoint(
            images=tiled_images,
            raw_images=tiled_raw_images,
            find_queries=tiled_find_queries,
            get_queries=tiled_get_queries,
        )

    def _subsample_queries(self, datapoint: Datapoint, max_query_num: int):
        """Subsample to keep at most `max_query_num` queries per frame in a datapoint."""
        # aggregate the find queries per stage
        num_frames = max(q.query_processing_order for q in datapoint.find_queries) + 1
        find_queries_per_stage = [[] for _ in range(num_frames)]
        for query in datapoint.find_queries:
            find_queries_per_stage[query.query_processing_order].append(query)

        # verify that all the stages have the same number of queries
        num_queries_per_stage = len(find_queries_per_stage[0])
        for queries in find_queries_per_stage:
            assert len(queries) == num_queries_per_stage
        if max_query_num <= 0 or num_queries_per_stage <= max_query_num:
            return datapoint

        # subsample the queries to keep only `max_query_num` queries
        sampled_inds = self.rng.sample(range(num_queries_per_stage), max_query_num)
        sampled_find_queries_per_stage = [
            [queries[idx] for idx in sampled_inds] for queries in find_queries_per_stage
        ]
        sampled_find_queries = sum(sampled_find_queries_per_stage, [])
        return Datapoint(
            images=datapoint.images,
            raw_images=datapoint.raw_images,
            find_queries=sampled_find_queries,
            get_queries=datapoint.get_queries,
        )
