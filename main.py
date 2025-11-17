from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader

# Load the dataset
dataset = OGB_MAG(root='OGBN‐MAG/')
print(dataset)
data = dataset[0]  # This is a HeteroData object
print(data)

loader = NeighborLoader(
    data,
    input_nodes=('paper', data['paper'].train_mask),
    num_neighbors=[30] * 2,
    batch_size=64,
)

sampled_data = next(iter(loader))
print(sampled_data['paper'].batch_size)
print(sampled_data['paper'].y)
