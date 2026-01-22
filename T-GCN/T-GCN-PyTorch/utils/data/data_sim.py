import argparse
import pickle
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class SpatioTemporalDataset(Dataset):
    def __init__(self, data_list, adj_list):
        """
        Args:
            data_list: 包含 data_x, data_y, data_y_mask 的列表
            adj_list: 包含所有样本邻接矩阵的列表
        """
        self.data_list = data_list
        self.adj_list = adj_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        
        # 1. 提取特征、标签和掩码
        # 数据维度通常为 (1, T, R, F)，使用 squeeze(0) 去掉第一维
        x = torch.FloatTensor(sample["data_x"]).squeeze(0)
        y = torch.FloatTensor(sample["data_y"]).squeeze(0)
        y_mask = torch.FloatTensor(sample["data_y_mask"]).squeeze(0)
        
        # 2. 从独立的 edges 列表中获取对应的邻接矩阵
        adj_data = self.adj_list[index]
        if hasattr(adj_data, "toarray"):
            # 如果是 scipy 稀疏矩阵
            adj = torch.FloatTensor(adj_data.toarray())
        else:
            # 如果已经是 numpy 数组
            adj = torch.FloatTensor(adj_data)
            
        return x, y, adj, y_mask

class SimSpatioTemporalPklDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,    # 仿真数据 pkl 文件路径 (data_x, data_y, data_y_mask)
        edges_path: str,   # 独立的邻接矩阵 pkl 文件路径
        batch_size: int = 32,
        split_ratio: float = 0.8,
        **kwargs
    ):
        super(SimSpatioTemporalPklDataModule, self).__init__()
        self._feat_path = feat_path
        self._edges_path = edges_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        
        # 加载数据和邻接矩阵
        self._load_data()

    def _load_data(self):
        # 加载特征数据
        with open(self._feat_path, 'rb') as f:
            self._full_data = pickle.load(f)
            
        # 加载邻接矩阵数据 (假设保存格式为一个包含所有样本 adj 的 List)
        # with open(self._edges_path, 'rb') as f:
        #     self._full_adjs = pickle.load(f)
        self._full_adjs = np.load(self._edges_path)
            
        if len(self._full_data) != len(self._full_adjs):
            print(f"警告: 特征样本数({len(self._full_data)})与邻接矩阵数({len(self._full_adjs)})不一致！")

        # 估算特征最大值 (用于兼容性或反归一化参考)
        sample_x = self._full_data[0]["data_x"]
        self._feat_max_val = np.max(sample_x)
        
        # 提取第一个样本的邻接矩阵供模型初始化时确定维度
        first_adj = self._full_adjs[0]
        self._adj = first_adj.toarray() if hasattr(first_adj, "toarray") else first_adj

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        # parser.add_argument("--feat_path", type=str, help="仿真特征数据路径")
        # parser.add_argument("--edges_path", type=str, help="邻接矩阵数据路径")
        return parser

    def setup(self, stage: str = None):
        # 这里保留你要求的 [:1000] 限制，也可以根据实际情况移除
        full_data = self._full_data
        full_adjs = self._full_adjs
        
        num_samples = len(full_data)
        split_idx = int(num_samples * self.split_ratio)

        if stage == "fit" or stage is None:
            self.train_dataset = SpatioTemporalDataset(
                full_data[:split_idx], full_adjs[:split_idx]
            )
            self.val_dataset = SpatioTemporalDataset(
                full_data[split_idx:], full_adjs[split_idx:]
            )
        
        if stage == "test":
            self.test_dataset = SpatioTemporalDataset(full_data, full_adjs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj