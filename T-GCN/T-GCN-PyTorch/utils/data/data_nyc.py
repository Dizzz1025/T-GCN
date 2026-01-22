import argparse
import pickle
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# 自定义 Dataset 以处理字典格式的数据包
class SpatioTemporalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        
        # 1. 提取特征和标签，并转为 Tensor
        # 数据在封装时是 (1, T, R, F)，这里去掉多余的 Batch 维度
        x = torch.FloatTensor(sample["data_x"]).squeeze(0)
        y = torch.FloatTensor(sample["data_y"]).squeeze(0)
        x = x[..., :2]
        # 2. 处理邻接矩阵 (Scipy 稀疏矩阵 -> 稠密张量)
        # 即使数据已经是 80x80，模型计算通常需要 Tensor 格式
        adj_sparse = sample["adj"]
        if adj_sparse is not None:
            # .toarray() 将稀疏矩阵转为 numpy 数组，再转为 Tensor
            adj = torch.FloatTensor(adj_sparse.toarray())
        else:
            adj = torch.zeros((x.shape[1], x.shape[1])) # 兜底逻辑
            
        return x, y, adj

class NYCSpatioTemporalPklDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,  # 归一化后带 adj 的 pkl 文件路径
        batch_size: int = 64,
        split_ratio: float = 0.8,
        **kwargs
    ):
        super(NYCSpatioTemporalPklDataModule, self).__init__()
        self._feat_path = feat_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        
        # 预加载数据以获取元信息
        self._load_data()

    def _load_data(self):
        with open(self._feat_path, 'rb') as f:
            # 现在的 data 是一个 List[Dict]，每个 Dict 包含 x, y, adj
            self._full_data = pickle.load(f)
        
        # 计算 feat_max_val (虽然数据已归一化，但为了兼容性保留此属性)
        # 从第一个样本中估算或设为 1.0 (因为已经 z-score 归一化了)
        sample_x = self._full_data[0]["data_x"]
        self._feat_max_val = np.max(sample_x)
        
        # 提取第一个样本的邻接矩阵作为默认 adj 属性 (供模型初始化使用)
        first_adj_sparse = self._full_data[0]["adj"]
        self._adj = first_adj_sparse.toarray() if first_adj_sparse is not None else None

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        return parser

    def setup(self, stage: str = None):
        self._full_data = self._full_data
        num_samples = len(self._full_data)
        split_idx = int(num_samples * self.split_ratio)

        # 使用自定义的 Dataset
        if stage == "fit" or stage is None:
            self.train_dataset = SpatioTemporalDataset(self._full_data[:split_idx])
            self.val_dataset = SpatioTemporalDataset(self._full_data[split_idx:])
        
        if stage == "test":
            self.test_dataset = SpatioTemporalDataset(self._full_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        """返回第一个样本的邻接矩阵，用于模型层初始化"""
        return self._adj