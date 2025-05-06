import sys

import torch
import torch.nn.functional as f
from tqdm import tqdm


def graph_cls_step(
        gnn_model,
        attn_model,
        cls_model,
        dataloader,
        padding_func,
        padding_target,
        optimizer,
        criterion_alg,
        criterion_cls,
        alpha,
        beta,
        device,
        epoch
):
    gnn_model.train()
    attn_model.train()
    cls_model.train()

    total_loss = 0
    total_alg_loss = 0
    total_cls_loss = 0

    correct_predictions = 0.0
    total_predictions = 0.0

    for graph in tqdm(
        iterable=dataloader,
        desc=f"Epoch {epoch}",
        unit="batch",
        file=sys.stdout
    ):
        optimizer.zero_grad()

        graph.to(device)

        adj_matrix = torch.sparse_coo_tensor(
            indices=graph.edge_index,
            values=torch.ones(graph.edge_index.shape[1]).to(device),
            size=(graph.num_nodes, graph.num_nodes)
        ).to_dense()
        left_singular, _, _ = torch.svd(adj_matrix)

        padded_left_singular, mask = padding_func(
            singular_matrix=left_singular,
            target_columns=padding_target
        )
        padded_left_singular = padded_left_singular.to(device)

        node_embs, _ = gnn_model(graph)
        node_embs = node_embs.to(device)

        reweighed_left_singular = attn_model(
            query=node_embs,
            keys=padded_left_singular,
            values=padded_left_singular
        )

        normalized_node_emb = f.normalize(node_embs, p=2)
        normalized_attended_left_singular = f.normalize(reweighed_left_singular, p=2)

        alg_loss = criterion_alg(
            normalized_node_emb,
            normalized_attended_left_singular
        )
        total_alg_loss += alpha * alg_loss

        one_hop_counts = torch.bincount(
            torch.cat((graph.edge_index[0], graph.edge_index[1])),
            minlength=graph.num_nodes
        )
        one_hop_counts.to(device)

        logits = cls_model(normalized_node_emb)
        cls_loss = criterion_cls(logits, one_hop_counts)
        total_cls_loss += beta * cls_loss

        loss = alg_loss + cls_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted_classes = torch.max(logits, 1)
        correct_predictions += torch.eq(predicted_classes, one_hop_counts).long().sum().item()
        total_predictions += one_hop_counts.size(0)

    average_loss = total_loss / len(dataloader)
    average_alg_loss = total_alg_loss / len(dataloader)
    average_cls_loss = total_cls_loss / len(dataloader)

    one_hop_acc = (correct_predictions / total_predictions) * 100

    return (average_loss,
            average_alg_loss,
            average_cls_loss,
            one_hop_acc
            )
