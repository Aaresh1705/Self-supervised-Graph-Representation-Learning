import torch

from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import NeighborLoader

import lib
from lib.model import SupervisedNodePredictions
from lib.model import GraphSAGE


if __name__ == '__main__':
    root_path = 'OGBN-MAG/'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = lib.dataset.load_data(root_path, transform=transform, preprocess=preprocess)

    target_type = "paper"
    data_inductive = lib.dataset.to_inductive(data.clone(), target_type)

    train_loader = NeighborLoader(
        data_inductive,
        input_nodes=(target_type, data_inductive[target_type].train_mask),
        num_neighbors=[15, 10],
        batch_size=1024,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=(target_type, data[target_type].val_mask),
        num_neighbors=[15, 10],
        batch_size=2048,
    )

    hidden_dim = 128  # common dim after projection (and GNN input dim)
    num_classes = int(data_inductive[target_type].y.max()) + 1

    # GNN expects the post-encoder dim as input
    model = GraphSAGE(in_channels=hidden_dim, hidden_channels=hidden_dim)
    model = to_hetero(model, data_inductive.metadata(), aggr='sum')
    model.num_classes = num_classes
    model.output_dim = hidden_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    model = model.to(device)

    pipeline = SupervisedNodePredictions(model=model, device=device, optimizer=None, target_type=target_type)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pipeline.classifier.parameters()), lr=0.003)
    pipeline.optimizer = optimizer

    for epoch in range(10):
        loss = pipeline.train(train_loader)
        acc = pipeline.test(val_loader)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
