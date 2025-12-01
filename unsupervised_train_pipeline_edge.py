import lib
from lib.model import make_gmae, Readout, get_x_dict, train_edge_readout, test_edge_readout
from lib.dataset import load_data
import torch

from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.utils import negative_sampling

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("loading data...")
transform = Compose([ToUndirected(merge=False)])
preprocess = 'metapath2vec'
data = lib.dataset.load_data("", transform=transform, preprocess=preprocess)

paper_train_mask = data["paper"].train_mask
paper_test_mask  = data["paper"].val_mask
edge_type = ("paper", "has_topic", "field_of_study")
edge_index = data[edge_type].edge_index   
paper_idx = edge_index[0]               
fos_idx = edge_index[1]              
train_edge_mask = paper_train_mask[paper_idx]
test_edge_mask = paper_test_mask[paper_idx]
train_edge_index = edge_index[:, train_edge_mask]
test_edge_index  = edge_index #[:, test_edge_mask] allowed to see all edges during inference 

model_type = "gmae" # or gae

print("loading model...")
encoder, _, _, _ = make_gmae() if model_type == "gmae" else make_gae()
encoder.load_state_dict(torch.load(model_type + "_encoder", map_location=device, weights_only=True))
encoder.to(device)
encoder.eval()

print("getting embeddings...")
x_dict = get_x_dict(data)
with torch.no_grad():
    z_dict = encoder(x_dict, data.edge_index_dict)
z_paper = z_dict["paper"].detach()            
z_fos   = z_dict["field_of_study"].detach()  
readout = Readout(1) # single out channel -- probability

def edge_index_to_loader(edge_index, z_paper, z_fos, batch_size=1024):
    pos_edge_index = edge_index
    num_pos = pos_edge_index.size(1)
    num_paper = z_paper.size(0)
    num_fos   = z_fos.size(0)
    neg_edge_index = negative_sampling(
        pos_edge_index,
        num_nodes=(num_paper, num_fos),
        num_neg_samples=num_pos,
    )
    z_src_pos = z_paper[pos_edge_index[0]]
    z_dst_pos = z_fos[pos_edge_index[1]]  
    z_src_neg = z_paper[neg_edge_index[0]]
    z_dst_neg = z_fos[neg_edge_index[1]]  
    z_src = torch.cat([z_src_pos, z_src_neg], dim=0)
    z_dst = torch.cat([z_dst_pos, z_dst_neg], dim=0)
    y = torch.cat([
        torch.ones(num_pos, dtype=torch.float32),
        torch.zeros(num_pos, dtype=torch.float32),
    ], dim=0)
    dataset = TensorDataset(z_src, z_dst, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("building edge datasets...")
train_loader = edge_index_to_loader(train_edge_index, z_paper, z_fos)
test_loader  = edge_index_to_loader(test_edge_index,  z_paper, z_fos)

print("training edge predictor...")
for epoch in range(2):
    loss = train_edge_readout(readout, train_loader)
    acc  = test_edge_readout(readout, test_loader)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")
