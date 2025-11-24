#!/usr/bin/env python3

import lib
from lib.model import train_node_readout, make_gae, make_gmae, Readout, test_node_readout, get_x_dict
from lib.dataset import load_data
import torch

from torch_geometric.transforms import Compose, ToUndirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = "gmae" # or gae

print("loading data...")
transform = Compose([ToUndirected(merge=False)])
preprocess = 'metapath2vec'
data = lib.dataset.load_data("", transform=transform, preprocess=preprocess).to_device()
num_classes = int(data["paper"].y.max()) + 1

train_data = data.subgraph({
    "paper": data["paper"].train_mask.nonzero(as_tuple=False).view(-1)
})
val_data = data.subgraph({
    "paper": data["paper"].val_mask.nonzero(as_tuple=False).view(-1)
})

print("loading model...")
encoder, _, _, _ = make_gmae() if model_type == "gmae" else make_gae()
encoder.load_state_dict(torch.load(model_type + "_encoder", map_location=device, weights_only=True))
encoder.to(device)
encoder.eval()
readout = Readout(num_classes).to(device)

def dataset_to_loader(d):
    x_dict = get_x_dict(d)
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    with torch.no_grad():
        z_dict = encoder(x_dict, edge_index_dict)
    z_paper = z_dict["paper"]
    z_paper = z_paper.detach().cpu()
    y_paper = d["paper"].y.cpu()
    torchdataset = torch.utils.data.TensorDataset(z_paper, y_paper)
    return torch.utils.data.DataLoader(
        torchdataset,
        batch_size=16,
        shuffle=True
    )

print("generating embeddings...")
train_loader = dataset_to_loader(train_data)
val_loader   = dataset_to_loader(val_data)
print("embeddings generated")

for epoch in range(10):
    loss = train_node_readout(readout, train_loader)
    acc  = test_node_readout(readout, val_loader)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
