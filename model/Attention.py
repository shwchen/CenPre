import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim

        self.W_q = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=False)
        self.W_k = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=False)
        self.W_v = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=False)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = f.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_weights, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, output_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.depth = feature_dim // num_heads

        assert self.depth * num_heads == self.feature_dim, "feature_dim must be divisible by num_heads"

        self.W_qs = nn.ModuleList([nn.Linear(feature_dim, self.depth, bias=False) for _ in range(num_heads)])
        self.W_ks = nn.ModuleList([nn.Linear(feature_dim, self.depth, bias=False) for _ in range(num_heads)])
        self.W_vs = nn.ModuleList([nn.Linear(feature_dim, self.depth, bias=False) for _ in range(num_heads)])

        self.final_linear = nn.Linear(in_features=feature_dim, out_features=output_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        q = torch.cat([w_q(x).unsqueeze(1) for w_q in self.W_qs], dim=1)
        k = torch.cat([w_k(x).unsqueeze(1) for w_k in self.W_ks], dim=1)
        v = torch.cat([w_v(x).unsqueeze(1) for w_v in self.W_vs], dim=1)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        attention_weights = f.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        concatenated_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1)
        return self.final_linear(concatenated_output)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim

        self.W_q = nn.Linear(
            in_features=query_dim, out_features=query_dim, bias=False
        )
        self.W_k = nn.Linear(
            in_features=key_value_dim, out_features=query_dim, bias=False
        )
        self.W_v = nn.Linear(
            in_features=key_value_dim, out_features=query_dim, bias=False
        )

    def forward(self, query, keys, values):
        q = self.W_q(query)
        k = self.W_k(keys)
        v = self.W_v(values)

        attention_scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / (self.query_dim ** 0.5)
        attention_weights = f.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_weights, v).squeeze(1)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.depth = query_dim // num_heads

        assert self.depth * num_heads == self.query_dim, "query_dim must be divisible by num_heads"

        self.W_qs = nn.ModuleList([nn.Linear(query_dim, self.depth, bias=False) for _ in range(num_heads)])
        self.W_ks = nn.ModuleList([nn.Linear(key_dim, self.depth, bias=False) for _ in range(num_heads)])
        self.W_vs = nn.ModuleList([nn.Linear(value_dim, self.depth, bias=False) for _ in range(num_heads)])

        # self.final_linear = nn.Linear(in_features=query_dim, out_features=value_dim)

    def forward(self, query, keys, values):
        batch_size = query.shape[0]

        q = torch.cat([w_q(query).unsqueeze(1) for w_q in self.W_qs], dim=1)
        k = torch.cat([w_k(keys).unsqueeze(1) for w_k in self.W_ks], dim=1)
        v = torch.cat([w_v(values).unsqueeze(1) for w_v in self.W_vs], dim=1)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        attention_weights = f.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        concatenated_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1)
        return concatenated_output
