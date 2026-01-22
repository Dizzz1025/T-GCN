import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state, laplacian):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        # concatenation = torch.cat((inputs, hidden_state), dim=2)
        # # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        # concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        # concatenation = concatenation.reshape(
        #     (num_nodes, (self._num_gru_units + 1) * batch_size)
        # )
        # # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        # a_times_concat = self.laplacian @ concatenation
        # # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        # a_times_concat = a_times_concat.reshape(
        #     (num_nodes, self._num_gru_units + 1, batch_size)
        # )
        # # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        # a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        # a_times_concat = a_times_concat.reshape(
        #     (batch_size * num_nodes, self._num_gru_units + 1)
        # )
        # # A[x, h]W + b (batch_size * num_nodes, output_dim)
        # outputs = a_times_concat @ self.weights + self.biases
        # # A[x, h]W + b (batch_size, num_nodes, output_dim)
        # outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # # A[x, h]W + b (batch_size, num_nodes * output_dim)
        # outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        # [x, h] -> (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        
        # 【核心修改】使用 torch.bmm 实现样本级动态图卷积
        # laplacian: (B, N, N)  concatenation: (B, N, Units+1)
        # A * [x, h] -> (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = torch.bmm(laplacian, concatenation)
        
        # 线性变换：(B * N, Units + 1) @ (Units + 1, Out_dim)
        a_times_concat = a_times_concat.reshape((-1, self._num_gru_units + 1))
        outputs = a_times_concat @ self.weights + self.biases
        
        # 格式还原
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._num_nodes = num_nodes
        self._hidden_dim = hidden_dim
        # self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state, laplacian):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state, laplacian))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, laplacian))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"num_nodes": self._num_nodes, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._num_nodes = num_nodes
        self._hidden_dim = hidden_dim
        # self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self._num_nodes, self._hidden_dim)

    def forward(self, inputs, laplacians):
        """
        inputs: (batch_size, seq_len, num_nodes)
        laplacians: (batch_size, num_nodes, num_nodes) 
                   或是 (batch_size, seq_len, num_nodes, num_nodes)
        """
        batch_size, seq_len, num_nodes = inputs.shape
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            # output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            # output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            # 判断 laplacians 是否随时间步变化
            if laplacians.dim() == 4:
                adj_t = laplacians[:, i, :, :]
            else:
                adj_t = laplacians # 整个样本序列共用一个动态矩阵
                
            _, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state, adj_t)
        
        output = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))  
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"num_nodes": self._num_nodes, "hidden_dim": self._hidden_dim}
