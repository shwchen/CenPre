import torch
from torch.optim import AdamW, lr_scheduler
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

import logging
import hydra
from hydra.core.hydra_config import HydraConfig

from model import GCN, GAT, GraphSAGE
from train import node_cls_step
from util import EarlyStopping, setup_seed, custom_data_split

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)


@hydra.main(config_path="_conf/", config_name="nodecls_config", version_base=None)
def main(cfg) -> None:
    setup_seed(cfg.exp.random_seed)
    device = f'cuda:{cfg.exp.gpu_num}' if torch.cuda.is_available() else 'cpu'

    data_name = f"og_{cfg.data.dataset}.pt" if cfg.exp.exp_name == "baseline" else f"{cfg.data.dataset}.pt"
    data = torch.load(f"{cfg.path.dataset_folder}/{cfg.data.dataset}/{data_name}")
    if not isinstance(data, Data):
        data = Data(**data[0])
    data.to(device)
    print(data)

    if cfg.exp.exp_name != "baseline":
        st_model = SentenceTransformer(
            model_name_or_path=cfg.path.st_model_folder
        )
        st_model.to(device)

        encoded_texts = st_model.encode(
            sentences=data.raw_texts,
            batch_size=cfg.model.encoding_batch_size,
            convert_to_tensor=True,
            device=device
        )
        data.x = torch.tensor(
            data=encoded_texts,
            dtype=torch.float32
        )

        del st_model
        torch.cuda.empty_cache()

    gnn_model_dict = {
        "gat": GAT,
        "gcn": GCN,
        "graphsage": GraphSAGE
    }
    gnn = gnn_model_dict[cfg.model.gnn_model_name](
        num_features=cfg.data.num_features if cfg.exp.exp_name == "baseline" else data.x.size(1),
        num_classes=max(data.y)+1
    )
    gnn = gnn.to(device)

    if cfg.data.split != -1:
        split = cfg.data.split
        masks = {
            "train": data.train_masks[split],
            "valid": data.val_masks[split],
            "test": data.test_masks[split]
        }
    else:
        train_mask, valid_mask, test_mask = custom_data_split(
            num_samples=cfg.data.num_nodes,
            train_percent=cfg.data.train_percent,
            val_percent=cfg.data.valid_percent
        )
        masks = {
            "train": train_mask, "valid": valid_mask, "test": test_mask
        }

    optimizer = AdamW(
        params=gnn.parameters(),
        lr=cfg.optim.learning_rate
    )
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=cfg.sched.patience
    )
    early_stopping = EarlyStopping(
        patience=cfg.es.patience,
        min_delta=cfg.es.min_delta,
        name=f"{cfg.exp.exp_name}-{cfg.exp.random_seed}",
        save_path=f"{cfg.path.ckpt_folder}/{cfg.data.dataset}",
        save_ckpt=cfg.exp.save_ckpt,
        save_st_model=False
    )

    for epoch in range(cfg.exp.max_epochs):
        train_loss, train_acc = node_cls_step(
            model=gnn,
            data=data,
            masks=masks,
            optimizer=optimizer,
            criterion=criterion,
            phase="train"
        )
        valid_loss, valid_acc = node_cls_step(
            model=gnn,
            data=data,
            masks=masks,
            optimizer=optimizer,
            criterion=criterion,
            phase="valid"
        )

        scheduler.step(metrics=valid_loss)
        early_stopping(
            val_loss=valid_loss,
            model=gnn
        )

        logging.info(
            f"Epoch {epoch}: t_loss {train_loss:.7f}, t_acc {train_acc:.2f}%; "
            f"v_loss {valid_loss:.7f}, v_acc {valid_acc:.2f}%; "
            f"lr {optimizer.param_groups[0]['lr']}, es {early_stopping.counter}."
        )

        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    logging.info(f"Best checkpoint saved at {cfg.path.ckpt_folder}")

    gnn.load_state_dict(
        state_dict=torch.load(
            f=f"{cfg.path.ckpt_folder}/{cfg.data.dataset}/{cfg.exp.exp_name}-{cfg.exp.random_seed}.pt"
        )
    )

    test_loss, test_acc = node_cls_step(
        model=gnn,
        data=data,
        masks=masks,
        optimizer=optimizer,
        criterion=criterion,
        phase="test"
    )
    logging.info(
        f"Training Completed.\n"
        f"Results: test_loss {test_loss:.7f}, test_acc {test_acc:.2f}%; "
    )
    print(f"Logs and related run info can be found at {HydraConfig.get().runtime.output_dir}")

    if test_acc > 90:
        with open("/workspace/research/TextStructureAlign/_log/test.txt", "a") as file:
            file.write(f"seed: {cfg.exp.random_seed}, score: {test_acc:.2f}\n")
        file.close()


if __name__ == "__main__":
    main()
