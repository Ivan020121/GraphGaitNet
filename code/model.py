import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, global_mean_pool, global_max_pool


class GaitGraphNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, model_type, gcn_layers, attn_head, encoder_layers, num_classes):
        super(GaitGraphNet, self).__init__()
        self.input_dim = input_dim  # 100
        self.output_dim = output_dim  # 128
        self.model_type = model_type  # ['GCNConv', 'GraphConv', 'GatedGraphConv']
        self.gcn_layers = gcn_layers  # [2, 3, 4]
        self.attn_head = attn_head  # [2, 4]
        self.encoder_layers = encoder_layers  # [2, 4]
        self.num_classes = num_classes  # [2, 4]

        self.gcn_layer = self.build_gcn_layers()
        encoder_layer = TransformerEncoderLayer(2 * output_dim * gcn_layers,
                                                dim_feedforward=4 * output_dim * gcn_layers,
                                                nhead=attn_head,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, encoder_layers)

        # 添加 Dropout 层
        self.predictor = nn.Sequential(
            nn.Linear(2 * output_dim * gcn_layers, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout
            nn.Linear(256, 2),
        )

        # Transformer Encoder 输入前的 Dropout
        self.transformer_dropout = nn.Dropout(0.2)

    def build_gcn_layers(self):
        gcn_layers = nn.ModuleList()

        for i in range(self.gcn_layers):
            input_dim = self.input_dim if i == 0 else self.output_dim  # 第一层用输入维度，后续层用输出维度
            if self.model_type == 'GCNConv':
                gcn_layers.append(GCNConv(input_dim, self.output_dim))
            elif self.model_type == 'GraphConv':
                gcn_layers.append(GraphConv(input_dim, self.output_dim))
            elif self.model_type == 'GatedGraphConv':
                gcn_layers.append(GatedGraphConv(self.output_dim, input_dim))  # GatedGraphConv 参数顺序不同
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

        return gcn_layers

    def positional_encoding(self, t, d, device='cpu'):
        pos = torch.arange(t, dtype=torch.float32, device=device).unsqueeze(1)  # (t, 1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) *
                             (-math.log(10000.0) / d))  # 每隔两个生成频率因子
        pe = torch.zeros(t, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数位置使用 sin
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数位置使用 cos
        return pe

    def build_combined_graph(self, x, edge_index, edge_weight):
        num_samples, num_time_steps, num_nodes, input_dim = x.size()
        x = x.view(-1, num_nodes, input_dim)

        num_edges = edge_index.size(1)
        edge_index = edge_index.repeat(1, num_samples * num_time_steps)
        offsets = torch.arange(0, num_samples * num_time_steps, device=edge_index.device).repeat_interleave(
            num_edges) * num_nodes
        edge_index += offsets

        edge_weight = edge_weight.repeat(num_samples * num_time_steps)

        x = x.view(-1, input_dim)
        batch = torch.arange(num_samples * num_time_steps, device=edge_index.device).repeat_interleave(num_nodes)

        return x, edge_index, edge_weight, batch

    def forward(self, x, edge_index, edge_weight, mask):
        num_samples, num_time_steps, num_nodes, input_dim = x.size()
        x, edge_index, edge_weight, batch = self.build_combined_graph(x, edge_index, edge_weight)

        readout_features = []
        for conv in self.gcn_layer:
            x_residual = x
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = x + x_residual
            x = F.dropout(x, p=0.2)
            mean_pool = global_mean_pool(x, batch)  # 平均池化
            max_pool = global_max_pool(x, batch)  # 最大池化
            readout_features.append(torch.cat([mean_pool, max_pool], dim=-1))

        x = torch.cat(readout_features, dim=-1)  # 拼接所有层的 readout
        hidden_dim = x.size(-1)
        x = x.view(num_samples, num_time_steps, hidden_dim)

        pe = self.positional_encoding(num_time_steps, hidden_dim, device=x.device)
        x = x + pe

        x = self.transformer_dropout(x)  # 添加 Transformer 前的 Dropout
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        x = torch.mean(x, dim=1)
        x = self.predictor(x)

        return x
