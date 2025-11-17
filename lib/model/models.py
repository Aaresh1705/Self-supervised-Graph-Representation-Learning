import torch
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

code_size = 16
node_property_classes = 349
embed_dim = 128

metadata = (['paper', 'author', 'institution', 'field_of_study'], [('author', 'affiliated_with', 'institution'), ('author', 'writes', 'paper'), ('paper', 'cites', 'paper'), ('paper', 'has_topic', 'field_of_study'), ('institution', 'rev_affiliated_with', 'author'), ('paper', 'rev_writes', 'author'), ('paper', 'rev_cites', 'paper'), ('field_of_study', 'rev_has_topic', 'paper')])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(p=0.5)  # avoid FX tracing warning

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x

def make_gae():
    gae_encoder = pyg.nn.to_hetero(pyg.nn.models.GraphSAGE(
        in_channels=-1, # lazy init for auto hetero
        hidden_channels=16,
        num_layers=2,
        out_channels=code_size,
    ).to(device), metadata)
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
        metadata,
    ).to(device)
    gmae_decoder = pyg.nn.models.MLP([code_size, 128]).to(device)
    mask_embedding = torch.nn.Parameter(torch.zeros(128, device=device))
    remask_embedding = torch.nn.Parameter(torch.zeros(code_size, device=device))
    return gmae_encoder, gmae_decoder, mask_embedding, remask_embedding

def get_x_dict(data, embeddings):
    return {node_type: data[node_type].x if "x" in data[node_type] else embeddings[node_type].weight for node_type in data.node_types}

class Readout(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.lin(x)

class Pretrained(torch.nn.Module):
    def __init__(self, encoder, code_size, num_classes):
        self.encoder = encoder
        self.readout = Readout(code_size, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.readout(z)
