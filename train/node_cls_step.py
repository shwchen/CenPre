import torch


def node_cls_step(
        model,
        data,
        masks,
        optimizer,
        criterion,
        phase
):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    mask = masks[phase]

    total_loss = 0.0
    total_pred = 0
    correct_pred = 0

    optimizer.zero_grad()
    with torch.set_grad_enabled(mode=is_train):
        out = model(data)
        loss = criterion(out[mask], data.y[mask])

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        _, predicted_classes = torch.max(out[mask], 1)
        total_pred += mask.sum().item()
        correct_pred += torch.eq(predicted_classes, data.y[mask]).long().sum().item()

    avg_loss = total_loss / len(data.y[mask])
    accuracy = 100 * correct_pred / total_pred

    return avg_loss, accuracy
