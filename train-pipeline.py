import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
dataset = OGB_MAG(root='OGBN-MAG_preprocess/', preprocess='metapath2vec', transform=ToUndirected())
data = dataset[0]
print(data)

# Target node type and feature/label info
target_type = 'paper'
num_features = data[target_type].x.size(-1)
num_classes = int(data[target_type].y.max()) + 1

# ------------------------------------------------------------
# 2. Define a GraphSAGE model (homogeneous first)
# ------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# ------------------------------------------------------------
# 3. Convert it to handle heterogeneous graphs
# ------------------------------------------------------------
# to_hetero() automatically builds separate SAGEConv layers for each relation type.
model = GraphSAGE(num_features, 128, num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ------------------------------------------------------------
# 4. Data loaders
# ------------------------------------------------------------
train_loader = NeighborLoader(
    data,
    input_nodes=(target_type, data[target_type].train_mask),
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

# ------------------------------------------------------------
# 5. Training setup
# ------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

def train():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out[target_type][:batch[target_type].batch_size]
        y = batch[target_type].y[:batch[target_type].batch_size].view(-1)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / len(train_loader)

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out[target_type][:batch[target_type].batch_size]
        y = batch[target_type].y[:batch[target_type].batch_size].view(-1)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == y).sum())
        total_examples += y.size(0)
    return total_correct / total_examples

# ------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------
for epoch in range(1, 6):
    loss = train()
    acc = test(val_loader)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
