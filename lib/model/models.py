import torch
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

code_size = 16
node_property_classes = 349

metadata = (['paper', 'author', 'institution', 'field_of_study'],
            [('author', 'affiliated_with', 'institution'),
             ('author', 'writes', 'paper'),
             ('paper', 'cites', 'paper'),
             ('paper', 'has_topic', 'field_of_study'),
             ('institution', 'rev_affiliated_with', 'author'),
             ('paper', 'rev_writes', 'author'),
             ('paper', 'rev_cites', 'paper'),
             ('field_of_study', 'rev_has_topic', 'paper')])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, num_classes=None):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(out_channels, out_channels))
        self.dropout = nn.Dropout(p=0.5)

        if num_classes:
            self.node_classifier = self._Classifier(out_channels, num_classes)
        self.edge_classifier = self._DotProductDecoder()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    class _Classifier(nn.Module):
        def __init__(self, output_dim, num_classes):
            super().__init__()
            self.head = nn.Linear(output_dim, num_classes)

        def forward(self, Z, node_idx):
            Z_batch = Z[node_idx]  # select the relevant nodes
            logits = self.head(Z_batch)  # shape [batch_size, num_classes]
            return logits

    class _DotProductDecoder(nn.Module):
        def forward(self, src_emb, dst_emb):
            logits = (src_emb * dst_emb).sum(dim=-1)
            return logits


def make_gae():
    gae_encoder = pyg.nn.to_hetero(pyg.nn.models.GraphSAGE(
        in_channels=-1, # lazy init for auto hetero
        hidden_channels=128,
        num_layers=2,
        out_channels=code_size,
    ).to(device), metadata)
    gae_decoder = pyg.nn.models.MLP([code_size, 128]).to(device)
    return gae_encoder, gae_decoder

def make_gmae():
    gmae_base_encoder = pyg.nn.models.GraphSAGE(
        in_channels=-1,  # lazy init
        hidden_channels=128,
        num_layers=2,
        out_channels=code_size,
    ).to(device)
    gmae_encoder = pyg.nn.to_hetero(
        gmae_base_encoder,
        metadata,
    ).to(device)
    gmae_decoder = pyg.nn.to_hetero(pyg.nn.models.GraphSAGE(
        in_channels = code_size,
        num_layers = 1,
        hidden_channels=128,
        out_channels=128
    ), metadata).to(device)
    mask_embedding = torch.nn.Parameter(torch.zeros(128, device=device))
    remask_embedding = torch.nn.Parameter(torch.zeros(code_size, device=device))
    return gmae_encoder, gmae_decoder, mask_embedding, remask_embedding

def get_x_dict(data):
    return {node_type: data[node_type].x for node_type in data.node_types}

class Readout(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lin = torch.nn.Linear(code_size, num_classes)

    def forward(self, x):
        return self.lin(x)

class EdgeReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(code_size * 2, 1)

    def forward(self, x):
        return self.lin(x)
