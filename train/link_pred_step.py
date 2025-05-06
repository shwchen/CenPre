import torch
import torch.nn.functional as f
from sklearn.metrics import roc_auc_score, average_precision_score


def link_pred_step(
        classifier,
        data,
        masks,
        optimizer,
        criterion,
        task,
        phase,
):
    is_train = phase == "train"
    classifier.train() if is_train else classifier.eval()

    mask = masks[phase]
    total_loss = 0.0
    total_pred = 0
    correct_pred = 0

    optimizer.zero_grad()
    with torch.set_grad_enabled(mode=is_train):
        if task == "link_cls":
            src_node_embs = data.x[data.edge_index[0]]
            des_node_embs = data.x[data.edge_index[1]]
        else:
            src_node_embs = data.x[data.link_prediction_pairs[:, 0]]
            des_node_embs = data.x[data.link_prediction_pairs[:, 1]]
        concat_emb = torch.cat(
            tensors=[src_node_embs, des_node_embs],
            dim=1
        )
        concat_emb = f.normalize(concat_emb, p=2, dim=1)

        logits = classifier(concat_emb)
        if task == "link_cls":
            labels = data.link_type_label.long()
        else:
            labels = f.one_hot(
                input=data.link_prediction_labels.long(),
                num_classes=2
            ).float()
        loss = criterion(logits[mask], labels[mask])

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if task == "link_cls":
            _, predicted_classes = torch.max(logits[mask], 1)
            total_pred += mask.sum().item()
            correct_pred += torch.eq(predicted_classes, data.link_type_label[mask]).long().sum().item()

    avg_loss = total_loss / len(data.link_prediction_labels[mask])

    probabilities = f.softmax(logits, dim=1)
    positive_class_probabilities = probabilities[mask][:, 1].detach().cpu().numpy()
    true_labels = data.link_prediction_labels[mask].cpu().numpy()
    roc_score = roc_auc_score(true_labels, positive_class_probabilities) * 100
    ap_score = average_precision_score(true_labels, positive_class_probabilities) * 100

    return avg_loss, roc_score, ap_score
