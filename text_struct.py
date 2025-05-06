import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from sentence_transformers import SentenceTransformer

import logging
import hydra
from hydra.core.hydra_config import HydraConfig

from data import prepare_graphtext_dataloader
from loss import ContrastiveAlignmentLoss, LabelSmoothingCrossEntropy
from model import SimpleClassifier, MultiHeadCrossAttention, MultiHeadSelfAttention
from util import EarlyStopping, count_parameters, setup_seed
from train import text_struct_step

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)


@hydra.main(config_path="_conf/", config_name="tsa_config", version_base=None)
def main(cfg) -> None:
    """Main function to run the training process."""
    setup_seed(cfg.exp.random_seed)
    device = torch.device(f"cuda:{cfg.exp.gpu_num}" if torch.cuda.is_available() else "cpu")

    st_model = SentenceTransformer(
        model_name_or_path=f"{cfg.path.pretrained_model_folder}/{cfg.model.pretrained_model_name}"
    )
    st_model.to(device)

    st_model_og = SentenceTransformer(
        model_name_or_path=f"{cfg.path.pretrained_model_folder}/{cfg.model.pretrained_model_name}"
    )
    st_model_og.to(device)

    for name, param in st_model.named_parameters():
        param.requires_grad = False
        if name.startswith(cfg.model.train_layer):
            param.requires_grad = True

    cls_model = SimpleClassifier(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        num_classes=cfg.data.num_classes
    )
    cls_model.to(device)

    if cfg.model.attn_model_name == "cross-attention":
        attn_model = MultiHeadCrossAttention(
            query_dim=768,
            key_dim=2708,
            value_dim=2708,
            num_heads=cfg.model.num_attn_heads
        )
    else:
        attn_model = MultiHeadSelfAttention(
            feature_dim=cfg.data.truncated_k,
            num_heads=cfg.model.num_attn_heads,
            output_dim=768
        )
    attn_model.to(device)

    st_total_params, st_trainable_params = count_parameters(model=st_model)
    cls_total_params, cls_trainable_params = count_parameters(model=cls_model)
    attn_total_params, attn_trainable_params = count_parameters(model=attn_model)
    trainable = st_trainable_params + cls_trainable_params + attn_trainable_params
    total = st_total_params + cls_total_params + attn_total_params

    logging.info(f"Total trainable parameters: {trainable}/{total}.")

    data = torch.load(
        f=f"{cfg.path.dataset_folder}/{cfg.data.dataset}/{cfg.data.dataset}.pt"
    )
    print(data)

    train_loader = prepare_graphtext_dataloader(
        texts=data.raw_texts,
        labels=data.one_hop_count,
        struct=data.reconstructed_adj_matrix,
        mask=None,
        batch_size=cfg.exp.batch_size
    )

    model_params = [param for param in st_model.parameters() if param.requires_grad]
    cls_params = list(cls_model.parameters())
    attn_params = list(attn_model.parameters())
    params_to_optimize = model_params + cls_params + attn_params

    optimizer = AdamW(params=params_to_optimize, lr=cfg.optim.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=cfg.sched.patience)
    early_stopping = EarlyStopping(
        patience=cfg.es.patience,
        min_delta=cfg.es.min_delta,
        name=f"{cfg.exp.exp_name}",
        save_path=f"{cfg.path.ckpt_folder}/{cfg.data.dataset}",
        save_ckpt=cfg.exp.save_ckpt,
        save_st_model=cfg.exp.save_st_model
    )

    criterion_dict = {
        "cl": ContrastiveAlignmentLoss(device=device),
        "lsce": LabelSmoothingCrossEntropy(smoothing=0.05),
        "mse": nn.MSELoss(),
        "cos": nn.CosineEmbeddingLoss(margin=0.9)
    }
    alg_criterion = criterion_dict[cfg.loss.alg_objective]
    cls_criterion = criterion_dict[cfg.loss.cls_objective]
    reg_criterion = criterion_dict[cfg.loss.reg_objective]

    for epoch in range(cfg.exp.max_epochs):
        train_loss, accuracy, avg_alg, avg_cls, avg_reg, avg_var = text_struct_step(
            st_model=st_model,
            og_st_model=st_model_og,
            classifier=cls_model,
            attention=attn_model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion_alg=alg_criterion,
            criterion_cls=cls_criterion,
            criterion_reg=reg_criterion,
            alpha=cfg.loss.alpha,
            beta=cfg.loss.beta,
            gamma=cfg.loss.gamma,
            device=device,
            phase='train',
            epoch=epoch
        )
        scheduler.step(train_loss)
        early_stopping(train_loss, st_model)

        logging.info(
            f"Epoch {epoch} Completed -> "
            f"loss: {train_loss:.4f} = alg {avg_alg:.4f} + cls {avg_cls:.4f} + reg {avg_reg:.4f}; "
            f"accuracy: {accuracy:.2f}%; "
            f"lr: {optimizer.param_groups[0]['lr']}; "
            f"es: {early_stopping.counter}; "
            f"var: {avg_var};"
        )

        with open("unaligned_vars.txt", "a") as f:
            f.write(str(avg_var)+"\n")
        f.close()

        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    if cfg.exp.save_ckpt:
        logging.info(
            f"Best ckpt scored {early_stopping.best_score:.4f} saved at {cfg.path.ckpt_folder}/{cfg.data.dataset}"
        )
    if cfg.exp.save_st_model:
        early_stopping.save_sentence_transformer(
            f"{cfg.path.pretrained_model_folder}/{cfg.model.pretrained_model_name}"
        )
        logging.info(
            f"The corresponding sentence transformer saved at {cfg.path.ckpt_folder}/{cfg.data.dataset}/st_models"
        )

    print(f"Logs and related run info can be found at {HydraConfig.get().runtime.output_dir}")

    torch.save(
        attn_model,
        "/workspace/tsa/_ckpt/cora/attn_model.pt"
    )


if __name__ == "__main__":
    main()
