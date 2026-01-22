import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 12,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        input_dim: int = 2,
        target_idx: int = 0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.feature_projection = nn.Linear(input_dim, 1)
        self.target_idx = target_idx
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
    def forward(self, x, adj):
        # (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes, F = x.size()
        x_flat = x.reshape(-1, F)
        x_projected = self.feature_projection(x_flat)
        x_input = x_projected.reshape(batch_size, seq_len, num_nodes)
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x_input, adj)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y, adj = batch
        num_nodes = x.size(2)
        predictions = self(x, adj)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y_target = y[:, :, :, self.target_idx].reshape((-1, num_nodes))
        return predictions, y_target

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        # predictions = predictions * self.feat_max_val
        # y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        # rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        # mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        # accuracy = utils.metrics.accuracy(predictions, y)
        # r2 = utils.metrics.r2(predictions, y)
        # explained_variance = utils.metrics.explained_variance(predictions, y)
        # metrics = {
        #     "val_loss": loss,
        #     "RMSE": rmse,
        #     "MAE": mae,
        #     "accuracy": accuracy,
        #     "R2": r2,
        #     "ExplainedVar": explained_variance,
        # }
        # self.log_dict(metrics)
        mse = self.val_mse(predictions, y)
        mae = self.val_mae(predictions, y)
        # 使用 self.log 记录对象，Lightning 会在 Epoch 结束时自动计算正确的全局平均值
        self.log("val_mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True)
        target_shape = (batch[1].size(0), batch[1].size(1), batch[1].size(2))
        return predictions.reshape(target_shape), y.reshape(target_shape)
    def test_step(self, batch, batch_idx):
        pass

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
        parser.add_argument("--loss", type=str, default="mse")
        return parser
