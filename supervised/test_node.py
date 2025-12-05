import sys, os

import torch

from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import NeighborLoader

# We had a problem importing the lib package when a file was inside a folder
# This fixed it
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from lib.dataset import load_data
from lib.model import SupervisedNodePredictions, GraphSAGE


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    root_path = "./"
    transform = Compose([ToUndirected(merge=False)])
    preprocess = "metapath2vec"

    root_path = './'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = load_data(root_path, transform=transform, preprocess=preprocess)

    target_type = "paper"

    test_loader = NeighborLoader(
        data,
        input_nodes=(target_type, data[target_type].test_mask),
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=False
    )

    hidden_dim = 128
    num_classes = int(data[target_type].y.max()) + 1

    model = GraphSAGE(in_channels=hidden_dim, out_channels=hidden_dim, num_classes=num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')
    model = model.to(device)

    best_model_path = "supervised/models/node/best.pt"
    print(f"Loading best model from: {best_model_path}")

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    pipeline = SupervisedNodePredictions(
        model=model,
        device=device,
        optimizer=None,
        target_type=target_type
    )

    test_acc = pipeline.test(test_loader)
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===\n")
