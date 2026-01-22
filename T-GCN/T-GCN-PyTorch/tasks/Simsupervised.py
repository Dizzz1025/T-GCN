import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

# 将类名修改为 SimSupervisedForecastTask 以防与 NYC 数据集的类名冲突
class SimSupervisedForecastTask(pl.LightningModule):
    """
    针对仿真数据优化的预测任务类。
    类名已更改为 SimSupervisedForecastTask 以避免与 NYC 任务冲突。
    """
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 59,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        input_dim: int = 2,
        target_idx: int = 0,
        **kwargs
    ):
        super(SimSupervisedForecastTask, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        
        # 自动从模型中获取隐藏层维度
        hidden_dim = (
            self.model.hyperparameters.get("hidden_dim")
            or self.model.hyperparameters.get("output_dim")
        )
        
        # 定义输出回归层
        self.regressor = (
            nn.Linear(hidden_dim, self.hparams.pre_len)
            if regressor == "linear" else regressor
        )
        
        self._loss_type = loss
        self.feat_max_val = feat_max_val
        self.feature_projection = nn.Linear(input_dim, 1)
        
        # 定义带掩码的评价指标
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x, adj):
        # x: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, f_dim = x.size()
        
        # 特征投影降维 (F -> 1)
        x_flat = x.reshape(-1, f_dim)
        x_projected = self.feature_projection(x_flat)
        x_input = x_projected.reshape(batch_size, seq_len, num_nodes)
        
        # 模型计算
        hidden = self.model(x_input, adj)
        hidden = hidden.reshape((-1, hidden.size(2)))
        
        # 回归预测
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
            
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions.transpose(1, 2)

    def shared_step(self, batch, batch_idx):
        # 适配仿真数据 batch (x, y, adj, y_mask)
        x, y, adj, y_mask = batch
        predictions = self(x, adj)
        
        y_target = y[:, :, :, self.hparams.target_idx]
        y_mask = y_mask[:, :, :, self.hparams.target_idx]
        
        return predictions, y_target, y_mask

    def masked_loss(self, preds, targets, mask):
        if self._loss_type == "mse":
            loss = F.mse_loss(preds, targets, reduction='none')
        elif self._loss_type == "mae":
            loss = F.l1_loss(preds, targets, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self._loss_type}")
            
        mask = mask.float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
            
        loss = loss * mask
        return torch.mean(loss)

    def training_step(self, batch, batch_idx):
        predictions, y, mask = self.shared_step(batch, batch_idx)
        loss = self.masked_loss(predictions, y, mask)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y, mask = self.shared_step(batch, batch_idx)
        loss = self.masked_loss(predictions, y, mask)
        
        valid_mask = mask > 0
        if valid_mask.any():
            valid_preds = predictions[valid_mask]
            valid_y = y[valid_mask]
            self.val_mse(valid_preds, valid_y)
            self.val_mae(valid_preds, valid_y)
            
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mse", self.val_mse, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_epoch=True)
        target_shape = (batch[1].size(0), batch[1].size(1), batch[1].size(2))
        return predictions.reshape(target_shape), y.reshape(target_shape)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"])
        parser.add_argument("--pre_len", type=int, default=12)
        return parser