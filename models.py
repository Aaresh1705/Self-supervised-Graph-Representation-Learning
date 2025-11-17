#!/usr/bin/env python3

# outdated from aksel

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from data import metadata

code_size = 16
node_property_classes = 349
embed_dim = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_gae():
    gae_encoder = pyg.nn.to_hetero(pyg.nn.models.GraphSAGE(
        in_channels=-1, # lazy init for auto hetero
        hidden_channels=16,
        num_layers=2,
        out_channels=code_size,
    ).to(device), metadata())
    gae_decoder = pyg.nn.models.MLP([code_size, 128]).to(device)
    return gae_encoder, gae_decoder

embed_dim = 128
def make_embeddings(dataset):
    node_embeddings = torch.nn.ModuleDict()
    for node_type in ["author", "institution", "field_of_study"]:
        node_embeddings[node_type] = torch.nn.Embedding(dataset.num_nodes_dict[node_type], embed_dim)
    return node_embeddings
        
def make_gmae():
    gmae_base_encoder = pyg.nn.models.GraphSAGE(
        in_channels=-1,  # lazy init
        hidden_channels=16,
        num_layers=2,
        out_channels=code_size,
    ).to(device)
    gmae_encoder = pyg.nn.to_hetero(
        gmae_base_encoder,
        dataset.metadata(),
    ).to(device)
    gmae_decoder = pyg.nn.models.MLP([code_size, 128]).to(device)
    mask_embedding = torch.nn.Parameter(torch.zeros(128, device=device))
    remask_embedding = torch.nn.Parameter(torch.zeros(code_size, device=device))
    return gmae_encode, gmae_decoder, mask_embedding, remask_embedding

def get_x_dict(data, embeddings):
    return {node_type: data[node_type].x if "x" in data[node_type] else embeddings[node_type].weight for node_type in data.node_types}

class Readout(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.lin(x)
