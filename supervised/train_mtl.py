import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import torch
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected, RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader

import lib
from lib.model import GraphSAGE
from lib.model import SupervisedMTL

import copy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # --------------------------------------------------
    # Load hetero MAG dataset
    # --------------------------------------------------
    root_path = '../'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'

    data = lib.dataset.load_data(root_path, transform=transform, preprocess=preprocess)

    target_node_type = "paper"
    target_edge_type = ('paper', 'has_topic', 'field_of_study')

    data_inductive = lib.dataset.to_inductive(copy.deepcopy(data), target_node_type)

    hidden_dim = data[target_node_type].x.size(-1)
    num_classes = int(data[target_node_type].y.max()) + 1

    # ------------------------------
    # Split edges by paper train/val
    # ------------------------------
    edge_index_full = data[target_edge_type].edge_index
    src_nodes = edge_index_full[0]

    edge_index_inductive = data_inductive[target_edge_type].edge_index
    src_ind = edge_index_inductive[0]
    train_edges_inductive = edge_index_inductive[:, data_inductive[target_node_type].train_mask[src_ind]]

    val_edges = edge_index_full[:, data[target_node_type].val_mask[src_nodes]]

    # ------------------------------
    # Build the model
    # ------------------------------
    base_model = GraphSAGE(
        in_channels=hidden_dim,
        out_channels=hidden_dim,
        num_layers=2,
        num_classes=num_classes
    )

    model = to_hetero(
        base_model,
        metadata=data_inductive.metadata(),
        aggr="sum"
    ).to(device)

    # -------------------------
    # Loaders
    # -------------------------
    train_loader = LinkNeighborLoader(
        data_inductive,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, train_edges_inductive),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=True
    )

    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, val_edges),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=False
    )

    # -------------------------
    # MTL pipeline
    # -------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),  # includes classifier + decoder + loss params
        lr=0.003,
        weight_decay=1e-5
    )

    trainer = SupervisedMTL(
        model=model,
        device=device,
        optimizer=optimizer,
        target_node_type=target_node_type,
        target_edge_type=target_edge_type
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_weighted_val = float("inf")
    save_path = 'supervised/models/edge/'
    for epoch in range(1, 11):
        train_metrics = trainer.train(train_loader)
        val_metrics = trainer.validate(val_loader)
        weighted_val = val_metrics["weighted_loss"]  # MUCH cleaner now

        print(f"Epoch {epoch:02d} | "
              f"Train Node Loss: {train_metrics['node_loss']:.4f} | "
              f"Train Edge Loss: {train_metrics['edge_loss']:.4f} | "
              f"Weighted Loss: {weighted_val:.4f} | "
              f"Val Node Acc: {val_metrics['node_acc']:.4f} | "
              f"Val Edge AUROC: {val_metrics['edge_auroc']:.4f} | "
              f"AUPRC: {val_metrics['edge_auprc']:.4f}")

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, save_path+f"checkpoint_{epoch}.pt")



        if weighted_val < best_weighted_val:
            best_weighted_val = weighted_val
            torch.save(model.state_dict(), save_path + "best.pt")


if __name__ == "__main__":
    main()
