import numpy as np
import random
import torch
from torch.backends import cudnn
from torch_geometric.utils import negative_sampling
from prettytable import PrettyTable


def count_parameters(model):
    total, trainable = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable


def custom_data_split(num_samples, train_percent=0.6, val_percent=0.2, shuffle=False):
    if shuffle:
        indices = torch.randperm(num_samples)
    else:
        indices = torch.arange(num_samples)

    train_size = int(num_samples * train_percent)
    val_size = int(num_samples * val_percent)

    train_mask = indices[:train_size]
    val_mask = indices[train_size:train_size + val_size]
    test_mask = indices[train_size + val_size:]

    train_mask = torch.zeros(num_samples, dtype=torch.bool).scatter_(0, train_mask, True)
    val_mask = torch.zeros(num_samples, dtype=torch.bool).scatter_(0, val_mask, True)
    test_mask = torch.zeros(num_samples, dtype=torch.bool).scatter_(0, test_mask, True)

    return train_mask, val_mask, test_mask


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True


def pretty_table(col_names, values, title):
    table = PrettyTable()
    table.title = title
    table.field_names = col_names
    table.add_row(values)
    return table


def pad_and_mask(singular_matrix, target_columns):
    num_nodes = singular_matrix.shape[0]
    current_columns = singular_matrix.shape[1]

    if current_columns >= target_columns:
        return singular_matrix[:, :target_columns], torch.ones(num_nodes, target_columns).bool()

    padded_matrix = torch.zeros(num_nodes, target_columns)
    padded_matrix[:, :current_columns] = singular_matrix

    mask = torch.zeros(num_nodes, target_columns).bool()
    mask[:, :current_columns] = True

    return padded_matrix, mask


def get_link_prediction_samples_and_labels(edge_index, num_nodes):
    edge_set = set([tuple(edge) for edge in edge_index.t().tolist()])

    positive_samples = torch.tensor(list(edge_set), dtype=torch.long).t()

    num_positive_samples = positive_samples.size(1)

    negative_samples = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_positive_samples)

    positive_labels = torch.ones(num_positive_samples, dtype=torch.long)
    negative_labels = torch.zeros(num_positive_samples, dtype=torch.long)

    samples = torch.cat([positive_samples, negative_samples], dim=1)
    labels = torch.cat([positive_labels, negative_labels])

    return samples, labels
