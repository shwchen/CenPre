import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch_geometric.datasets import Amazon, TUDataset
from torch_geometric.loader import DataLoader

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import logging
import hydra
from hydra.core.hydra_config import HydraConfig

from loss import ContrastiveAlignmentLoss
from model import (GAT, GCN, GIN, GraphSAGE,
                   MultiHeadCrossAttention, SimpleClassifier)
from train import graph_cls_step
from util import EarlyStopping, pad_and_mask, setup_seed


@hydra.main(config_path="_conf/", config_name="graphcls_config", version_base=None)
def main(cfg):
    setup_seed(cfg.exp.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TUDataset(
        root=cfg.path.dataset_folder,
        name=cfg.data.dataset
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    gnn_model_dict = {
        "gat": GAT,
        "gcn": GCN,
        "gin": GIN,
        "graphsage": GraphSAGE
    }
    gnn = gnn_model_dict[cfg.model.gnn_model_name](
        in_channels=dataset.num_features,
        embedding_dim=cfg.model.gnn_embedding_dim
    )
    gnn.to(device)

    attn = MultiHeadCrossAttention(
        query_dim=cfg.model.gnn_embedding_dim,
        key_dim=cfg.model.gnn_embedding_dim,
        value_dim=cfg.model.gnn_embedding_dim,
        num_heads=cfg.model.num_attn_heads
    )
    attn.to(device)

    classifier = SimpleClassifier(
        embedding_dim=cfg.model.gnn_embedding_dim,
        num_classes=cfg.data.num_one_hop_classes,
    )
    classifier.to(device)

    total_params = list(gnn.parameters()) + list(attn.parameters()) + list(classifier.parameters())
    optimizer = AdamW(
        params=total_params,
        lr=cfg.optim.learning_rate,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, patience=cfg.sched.patience
    )
    early_stopping = EarlyStopping(
        patience=cfg.es.patience,
        min_delta=cfg.es.min_delta,
        name=f"{cfg.exp.exp_name}",
        save_path=f"{cfg.path.ckpt_folder}/{cfg.data.dataset}",
        save_ckpt=cfg.exp.save_ckpt,
        save_st_model=cfg.exp.save_st_model
    )

    struct_align_loss = ContrastiveAlignmentLoss(device=device)
    neighbor_count_loss = nn.CrossEntropyLoss()

    for epoch in range(cfg.exp.max_epochs):
        loss, alg_loss, cls_loss, one_hop_acc = graph_cls_step(
            gnn_model=gnn,
            attn_model=attn,
            cls_model=classifier,
            dataloader=dataloader,
            padding_func=pad_and_mask,
            padding_target=cfg.model.gnn_embedding_dim,
            optimizer=optimizer,
            criterion_alg=struct_align_loss,
            criterion_cls=neighbor_count_loss,
            alpha=cfg.model.alpha,
            beta=cfg.model.beta,
            device=device,
            epoch=epoch
        )
        scheduler.step(loss)
        early_stopping(loss, gnn)

        logging.info(
            f"Epoch {epoch} - "
            f"loss: {loss:.4f} = alg {alg_loss:.6f}; "
            f"cls {cls_loss:.6f} + "
            f"1-hop acc: {one_hop_acc:.2f}; "
        )

    best_model = gnn_model_dict[cfg.model.gnn_model_name](
        in_channels=dataset.num_features,
        embedding_dim=cfg.model.gnn_embedding_dim
    )
    best_model.load_state_dict(
        state_dict=torch.load(f"{cfg.path.ckpt_folder}/{cfg.data.dataset}/{cfg.exp.exp_name}.pt")
    )

    graph_embs = np.array([best_model(graph)[1][0].detach().numpy() for graph in dataloader])
    graph_y = np.array([graph.y[0] for graph in dataloader])

    x_train, x_test, y_train, y_test = train_test_split(
        graph_embs, graph_y,
        stratify=graph_y,
        test_size=0.2,
        shuffle=True,
        random_state=cfg.exp.random_seed
    )

    clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    logging.info(f"SVM acc: {accuracy:.2f}%")

    print(f"Logs and related run info can be found at {HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
