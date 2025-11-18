import torch

from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import LinkNeighborLoader

import lib
from lib.model import SupervisedEdgePredictions
from lib.model import GraphSAGE


if __name__ == '__main__':
    root_path = 'OGBN-MAG/'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = lib.dataset.load_data(root_path, transform=transform, preprocess=preprocess)

    target_type = "paper"
    data_inductive = lib.dataset.to_inductive(data.clone(), target_type)

    target_edge_type = ('paper', 'has_topic', 'field_of_study')

    train_loader = LinkNeighborLoader(
        data_inductive,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, data_inductive[target_edge_type].edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, data[target_edge_type].edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=False,
    )

    hidden_dim = 128  # common dim after projection (and GNN input dim)

    # GNN expects the post-encoder dim as input
    model = GraphSAGE(in_channels=hidden_dim, hidden_channels=hidden_dim)
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

    for epoch in range(10):
        loss = pipeline.train(train_loader)
        acc, auroc, auprc = pipeline.test(val_loader)

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Acc: {acc:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
