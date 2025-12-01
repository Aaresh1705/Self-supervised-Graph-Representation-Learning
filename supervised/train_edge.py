import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import torch

from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import LinkNeighborLoader

import lib
from lib.model import SupervisedEdgePredictions
from lib.model import GraphSAGE

import copy



if __name__ == '__main__':
    root_path = '../'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = lib.dataset.load_data(root_path, transform=transform, preprocess=preprocess)

    target_type = "paper"
    data_inductive = lib.dataset.to_inductive(copy.deepcopy(data), target_type)

    target_edge_type = ('paper', 'has_topic', 'field_of_study')

    train_loader = LinkNeighborLoader(
        data_inductive,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, data_inductive[target_edge_type].edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=True,
    )
    edge_index_all = data[target_edge_type].edge_index
    src_papers = edge_index_all[0]

    # Use paper's val_mask to select *validation edges*:
    val_edge_mask = data['paper'].val_mask[src_papers]
    val_edge_index = edge_index_all[:, val_edge_mask]

    val_loader = LinkNeighborLoader(
        data,  # full graph for evaluation
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, val_edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=False,
    )

    hidden_dim = 128  # common dim after projection (and GNN input dim)

    # GNN expects the post-encoder dim as input
    model = GraphSAGE(in_channels=hidden_dim, out_channels=hidden_dim)
    model = to_hetero(model, data_inductive.metadata(), aggr='sum')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    pipeline = SupervisedEdgePredictions(
        model=model,
        device=device,
        optimizer=optimizer,
        target_edge_type=target_edge_type
    )

    best_acc = 0
    save_path = 'supervised/models/edge/'
    for epoch in range(10):
        loss = pipeline.train(train_loader)
        metrics = pipeline.test(val_loader)

        auprc = metrics["auprc"]

        print(
            f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"AUROC: {metrics['auroc']:.4f} | "
            f"AUPRC: {auprc:.4f}"
        )

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, save_path+f"checkpoint_{epoch}.pt")

        if auprc > best_acc:
            best_acc = auprc
            torch.save(model.state_dict(), save_path+"best.pt")