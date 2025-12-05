import sys, os
import copy

import torch

from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import NeighborLoader

# We had a problem importing the lib package when a file was inside a folder
# This fixed it
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from lib.dataset import load_data, to_inductive
from lib.model import SupervisedNodePredictions, GraphSAGE


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    root_path = './'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = load_data(root_path, transform=transform, preprocess=preprocess)

    target_type = "paper"
    data_inductive = to_inductive(copy.deepcopy(data), target_type)

    train_loader = NeighborLoader(
        data_inductive,
        input_nodes=(target_type, data_inductive[target_type].train_mask),
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=(target_type, data[target_type].val_mask),
        num_neighbors=[15, 10],
        batch_size=2048,
    )

    hidden_dim = 128
    num_classes = int(data_inductive[target_type].y.max()) + 1

    model = GraphSAGE(in_channels=hidden_dim, out_channels=hidden_dim, num_classes=num_classes)
    model = to_hetero(model, data_inductive.metadata(), aggr='sum')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    pipeline = SupervisedNodePredictions(
        model=model,
        device=device,
        optimizer=optimizer,
        target_type=target_type
    )

    best_acc = 0
    save_path = 'supervised/models/node/'
    for epoch in range(10):
        loss = pipeline.train(train_loader)
        acc = pipeline.test(val_loader)

        print(f"Epoch {epoch:02d} | "
              f"Loss: {loss:.4f} | "
              f"Val Acc: {acc:.4f}")

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, save_path+f"checkpoint_{epoch}.pt")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path+"best.pt")
