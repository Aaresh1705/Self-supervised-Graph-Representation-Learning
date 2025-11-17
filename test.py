import torch.nn as nn
import torch
from torch_geometric.nn import to_hetero
import torch.nn.functional as F
from torch_geometric.transforms import Compose, ToUndirected


class HeteroFeatureEncoder(nn.Module):
    """
    Per-node-type linear projection so all node types end up with the same dim.
    Pass an in_channels dict like: {'paper':128, 'author':256, ...}
    """
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()
        self.encoders = nn.ModuleDict({
            nt: nn.Linear(in_ch, out_channels) for nt, in_ch in in_channels_dict.items()
        })

    def forward(self, x_dict):
        # IMPORTANT: use per-key access (no zipping which can scramble dict order)
        return {nt: F.relu(self.encoders[nt](x)) for nt, x in x_dict.items()}


if __name__ == '__main__':
    from torch_geometric.loader import NeighborLoader
    import lib
    from lib.model import train, test
    from lib.model import GraphSAGE_test

    root_path = 'OGBN-MAG/'
    transform = Compose([ToUndirected()])
    preprocess = 'metapath2vec'
    data = lib.dataset.load_data(root_path, transform=transform, preprocess=preprocess)

    target_type = "paper"
    data_inductive = lib.dataset.to_inductive(data.clone(), target_type)

    # Build per-type input dims for the encoder
    in_channels_dict = {nt: data_inductive[nt].x.size(-1) for nt in data_inductive.node_types}
    hidden_dim = 128  # common dim after projection (and GNN input dim)
    num_classes = int(data_inductive[target_type].y.max()) + 1

    train_loader = NeighborLoader(
        data_inductive,
        input_nodes=(target_type, data_inductive[target_type].train_mask),
        num_neighbors=[15, 10],
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=(target_type, data[target_type].val_mask),
        num_neighbors=[15, 10],
        batch_size=2048,
        num_workers=0,
    )

    # GNN expects the post-encoder dim as input
    model = GraphSAGE_test(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=num_classes)
    model = to_hetero(model, data_inductive.metadata(), aggr='sum')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Per-type encoder projecting to hidden_dim
    feature_encoder = HeteroFeatureEncoder(in_channels_dict, out_channels=hidden_dim).to(device)

    for epoch in range(1, 6):
        loss = train(model, device, optimizer, train_loader, feature_encoder, target_type)
        acc = test(model, device, val_loader, feature_encoder, target_type)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
