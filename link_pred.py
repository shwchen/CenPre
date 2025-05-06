import torch
from torch.optim import AdamW, lr_scheduler
from sentence_transformers import SentenceTransformer

import logging
import hydra
from hydra.core.hydra_config import HydraConfig

from model import SimpleClassifier
from train import link_pred_step
from util import EarlyStopping, setup_seed, custom_data_split

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)


@hydra.main(config_path="_conf/", config_name="linkpred_config", version_base=None)
def main(cfg) -> None:
    setup_seed(cfg.exp.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_name = f"{cfg.data.dataset}.pt"
    data = torch.load(f"{cfg.path.dataset_folder}/{cfg.data.dataset}/{data_name}")
    data.to(device)

    if cfg.exp.exp_name != "baseline":
        st_model = SentenceTransformer(
            model_name_or_path=cfg.path.st_model_folder
        )
        st_model.to(device)

        encoded_texts = st_model.encode(
            sentences=data.raw_texts,
            batch_size=cfg.model.encoding_batch_size,
            convert_to_tensor=True,
            device="cuda"
        )
        data.x = torch.tensor(
            data=encoded_texts,
            dtype=torch.float32
        )

        del st_model
        torch.cuda.empty_cache()

    classifier = SimpleClassifier(
        embedding_dim=data.x.size(1)*2,
        num_classes=2
    )
    classifier.to(device)

    train_mask, valid_mask, test_mask = custom_data_split(
        num_samples=data.link_prediction_pairs.shape[0],
        train_percent=cfg.data.train_percent,
        val_percent=cfg.data.valid_percent,
        shuffle=True
    )
    masks = {
        "train": train_mask, "valid": valid_mask, "test": test_mask
    }

    optimizer = AdamW(
        params=list(classifier.parameters()),
        lr=cfg.optim.learning_rate
    )
    criterion = torch.nn.BCEWithLogitsLoss() if cfg.exp.exp_name != "link_cls" else torch.nn.CrossEntropyLoss()

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=cfg.sched.patience
    )
    early_stopping = EarlyStopping(
        patience=cfg.es.patience,
        min_delta=cfg.es.min_delta,
        name=f"{cfg.exp.exp_name}",
        save_path=f"{cfg.path.ckpt_folder}/{cfg.data.dataset}",
        save_ckpt=cfg.exp.save_ckpt,
        save_st_model=False
    )

    for epoch in range(cfg.exp.max_epochs):
        train_loss, train_auc, train_ap = link_pred_step(
            classifier=classifier,
            data=data,
            masks=masks,
            optimizer=optimizer,
            criterion=criterion,
            task=cfg.exp.exp_name,
            phase="train"
        )
        valid_loss, valid_auc, valid_ap = link_pred_step(
            classifier=classifier,
            data=data,
            masks=masks,
            optimizer=optimizer,
            criterion=criterion,
            task=cfg.exp.exp_name,
            phase="valid"
        )

        scheduler.step(metrics=valid_loss)
        early_stopping(
            val_loss=valid_loss,
            model=classifier
        )

        logging.info(
            f"Epoch {epoch}: t_loss {train_loss:.7f}, t_auc {train_auc:.2f}%, t_ap {train_ap:.2f}%; "
            f"v_loss {valid_loss:.7f}, v_auc {valid_auc:.2f}%, v_ap {valid_ap:.2f}%; "
            f"lr {optimizer.param_groups[0]['lr']}, es {early_stopping.counter}."
        )

        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    logging.info(f"Best checkpoint saved at {cfg.path.ckpt_folder}")

    classifier.load_state_dict(
        state_dict=torch.load(
            f=f"{cfg.path.ckpt_folder}/{cfg.data.dataset}/{cfg.exp.exp_name}.pt"
        )
    )

    test_loss, test_auc, test_ap = link_pred_step(
        classifier=classifier,
        data=data,
        masks=masks,
        optimizer=optimizer,
        criterion=criterion,
        task=cfg.exp.exp_name,
        phase="test"
    )
    logging.info(
        f"Training Completed.\n"
        f"Results: test_loss {test_loss:.7f}, test_auc {test_auc:.2f}%, test_ap {test_ap:.2f}%."
    )
    print(f"Logs and related run info can be found at {HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
