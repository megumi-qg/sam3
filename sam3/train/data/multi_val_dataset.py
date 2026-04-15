# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Dict, Iterable, Iterator


class MultiValLoader:
    """
    包装多个验证数据集的 DataLoader，提供迭代和长度接口
    这个类可以直接被训练器使用，不需要额外的 DataLoader 包装
    """
    
    def __init__(self, val_datasets: Dict[str, any], epoch: int = 0):
        """
        Args:
            val_datasets: 字典，键是数据集名称（如 "acdc", "camus"），值是 TorchDataset 对象
            epoch: 当前 epoch
        """
        self.val_datasets = val_datasets
        self.keys = list(val_datasets.keys())
        self.epoch = epoch
        self._total_length = None
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch"""
        self.epoch = epoch
        # 也设置子数据集的 epoch
        for dataset in self.val_datasets.values():
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            if hasattr(dataset.dataset, "set_epoch"):
                dataset.dataset.set_epoch(epoch)
    
    def __iter__(self) -> Iterator[Dict]:
        """
        返回一个迭代器，依次遍历所有验证数据集
        每次迭代返回 collate_fn 的结果，格式为 {dict_key: BatchedDatapoint}
        collate_fn 已经将 dict_key 设置为对应的数据集名称（如 "acdc", "camus"）
        """
        for key in self.keys:
            dataset = self.val_datasets[key]
            loader = dataset.get_loader(self.epoch)
            for batch in loader:
                # batch 已经是 {dict_key: BatchedDatapoint} 格式
                # 直接返回，不需要再包装
                yield batch
    
    def __len__(self):
        """返回所有验证数据集的总 batch 数"""
        if self._total_length is None:
            # 计算总长度：每个验证数据集的 batch 数之和
            total = 0
            for dataset in self.val_datasets.values():
                loader = dataset.get_loader(self.epoch)
                total += len(loader)
            self._total_length = total
        return self._total_length


class MultiValDataset:
    """
    包装多个验证数据集，提供 get_loader 接口供训练器使用
    """
    
    def __init__(self, val_datasets: Dict[str, any]):
        """
        Args:
            val_datasets: 字典，键是数据集名称（如 "acdc", "camus"），值是 TorchDataset 对象
        """
        self.val_datasets = val_datasets
        self.keys = list(val_datasets.keys())
        self.epoch = 0
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch"""
        self.epoch = epoch
        # 也设置子数据集的 epoch
        for dataset in self.val_datasets.values():
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            if hasattr(dataset.dataset, "set_epoch"):
                dataset.dataset.set_epoch(epoch)
    
    def get_loader(self, epoch: int) -> Iterable:
        """
        返回一个 MultiValLoader 对象，支持迭代和 len() 方法
        每次迭代返回一个字典，键是数据集名称，值是 batch
        """
        self.set_epoch(epoch)
        # 返回 MultiValLoader，它实现了 __iter__ 和 __len__ 方法
        return MultiValLoader(self.val_datasets, epoch=epoch)

