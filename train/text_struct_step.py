import sys

import torch
import torch.nn.functional as f
from tqdm import tqdm


def text_struct_step(
        st_model,
        og_st_model,
        classifier,
        attention,
        data_loader,
        optimizer,
        criterion_alg,
        criterion_cls,
        criterion_reg,
        alpha,
        beta,
        gamma,
        device,
        phase,
        epoch
):
    is_train = phase == 'train'
    st_model.train() if is_train else st_model.eval()
    classifier.train() if is_train else classifier.eval()

    total_loss = 0.0
    total_alg_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    correct_predictions = 0.0
    total_predictions = 0.0
    vars = 0.0

    for x, labels, struct_repr in tqdm(
            iterable=data_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            file=sys.stdout
    ):
        if isinstance(x, torch.Tensor):
            x.to(device)
        labels = labels.to(device)
        struct_repr = struct_repr.to(device)

        if is_train:
            optimizer.zero_grad()

        with ((torch.set_grad_enabled(mode=is_train))):
            encoded_texts = st_model.tokenize(x)

            for key in encoded_texts.keys():
                if isinstance(encoded_texts[key], torch.Tensor):
                    encoded_texts[key] = encoded_texts[key].to(device)

            og_st_embeddings = og_st_model.forward(encoded_texts)["sentence_embedding"]
            og_st_embeddings = f.normalize(input=og_st_embeddings, p=2)

            ft_st_embeddings = st_model.forward(encoded_texts)["sentence_embedding"]
            ft_st_embeddings = f.normalize(input=ft_st_embeddings, p=2)

            vars += og_st_embeddings.mean(dim=0).norm().item()

            attn_output = attention(
                query=ft_st_embeddings,
                keys=struct_repr,
                values=struct_repr
            )
            # attn_output = attention(x=struct_repr)
            attn_output = f.normalize(attn_output, p=2)

            alg_loss = alpha * criterion_alg(ft_st_embeddings, attn_output)
            total_alg_loss += alg_loss

            logits = classifier(ft_st_embeddings)
            cls_loss = beta * criterion_cls(logits, labels)
            total_cls_loss += cls_loss

            reg_loss = gamma * criterion_reg(
                ft_st_embeddings,
                og_st_embeddings,
                target=torch.full(
                    size=(ft_st_embeddings.shape[0],),
                    fill_value=1
                ).to(device)
            )
            total_reg_loss += reg_loss

            loss = alg_loss + cls_loss + reg_loss
            total_loss += loss.item()

            if is_train:
                loss.backward()
                optimizer.step()

            _, predicted_classes = torch.max(logits, 1)
            correct_predictions += torch.eq(predicted_classes, labels).long().sum().item()
            total_predictions += labels.size(0)

    average_alg_loss = total_alg_loss / len(data_loader.dataset)
    average_cls_loss = total_cls_loss / len(data_loader.dataset)
    average_reg_loss = total_reg_loss / len(data_loader.dataset)

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = (correct_predictions / total_predictions) * 100

    return average_loss, accuracy, average_alg_loss, average_cls_loss, average_reg_loss, vars/170,
