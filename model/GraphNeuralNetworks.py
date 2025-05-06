import torch
import torch.nn.functional as f
import torch_geometric.nn as nn
from torch_geometric.nn.conv import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch_geometric.nn.pool import global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, heads=12, dropout_rate=0.2):
        super(GAT, self).__init__()
        self.heads = heads
        self.dropout_rate = dropout_rate

        self.conv1 = GATConv(
            in_channels=num_features, out_channels=256, heads=heads, dropout=dropout_rate
        )
        self.norm1 = LayerNorm(256 * heads)
        self.conv2 = GATConv(
            in_channels=256 * heads, out_channels=128, heads=heads, dropout=dropout_rate
        )
        self.norm2 = LayerNorm(128 * heads)
        self.conv3 = GATConv(
            in_channels=128 * heads, out_channels=num_classes, heads=1, concat=False, dropout=dropout_rate
        )
        self.norm3 = LayerNorm(num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = f.dropout(input=x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = f.leaky_relu(x)

        x = f.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = f.leaky_relu(x)

        x = f.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)

        return f.log_softmax(x, dim=1)


# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels=num_features, out_channels=384)
#         self.conv2 = GCNConv(in_channels=384, out_channels=num_classes)
#         self.norm1 = LayerNorm(in_channels=384)
#         self.norm2 = LayerNorm(in_channels=num_classes)
#
#     def forward(self, data, return_emb=False):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = self.norm1(x)
#         x = f.leaky_relu(x)
#         x = f.dropout(x, training=self.training)
#
#         x = self.conv2(x, edge_index)
#         x = self.norm2(x)
#
#         if return_emb:
#             return x
#         else:
#             return f.log_softmax(x, dim=1)
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=768, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

        self.norm1 = BatchNorm(hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        self.norm3 = LayerNorm(num_classes)

        self.dropout_rate = dropout_rate

    def forward(self, data, return_emb=False):
        x, edge_index = data.x, data.edge_index

        # First convolution layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout_rate, training=self.training)

        # Second convolution layer
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout_rate, training=self.training)

        # Third convolution layer
        x = self.conv3(x, edge_index)
        x = self.norm3(x)

        if return_emb:
            return x
        else:
            return f.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(GIN, self).__init__()
        nn1 = nn.MLP([in_channels, 32, 64])
        nn2 = nn.MLP([64, 64, embedding_dim])
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = self.conv2(x, edge_index)
        x = f.relu(x)

        batch = torch.zeros(x.size(0), dtype=torch.long)
        return x, global_mean_pool(x, batch)


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=num_features, out_channels=256)
        self.conv2 = SAGEConv(in_channels=256, out_channels=128)
        self.conv3 = SAGEConv(in_channels=128, out_channels=num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = f.leaky_relu(x)
        x = f.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = f.leaky_relu(x)
        x = f.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        return f.log_softmax(x, dim=1)
