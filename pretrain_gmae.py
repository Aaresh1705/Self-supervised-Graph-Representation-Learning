#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from data import load_dataset
from models import make_gmae, get_x_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

code_size = 16
node_property_classes = 349
gamma = 1
gmae_mask_rate = .5

dataset = load_dataset(remove_test=True)

def sce_loss(true, pred, mask, eps=1e-8):
    true = true[mask]
    pred = pred[mask]
    true_n = true / (true.norm(dim=-1, keepdim=True) + eps)
    pred_n = pred / (pred.norm(dim=-1, keepdim=True) + eps)
    loss = (1.0 - (true_n * pred_n).sum(dim=-1)).clamp(min=0) ** gamma
    return loss.mean()

def train_gmae(encoder, decoder, mask_embedding, remask_embedding, node_embeddings, data):
    optimizer = torch.optim.Adam(  list(encoder.parameters())
                                 + list(decoder.parameters())
                                 + list(node_embeddings.parameters())
                                 + [mask_embedding, remask_embedding],
                                 lr=0.01)
    data = data.to(device)
    paper_x = data["paper"].x.to(device)
    num_paper = data["paper"].num_nodes
    for epoch in range(2):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        data = data.to(device)
        x_dict = get_x_dict(data, node_embeddings)
        mask = (torch.rand(num_paper, device=device) < gmae_mask_rate)
        x_paper = x_dict["paper"].clone()
        x_paper[mask] = mask_embedding
        x_dict["paper"] = x_paper
        z_dict = encoder(x_dict, data.edge_index_dict)
        z_paper_masked = z_dict["paper"].clone()
        z_paper_masked[mask] = remask_embedding
        pred = decoder(z_paper_masked)
        loss = sce_loss(paper_x, pred, mask)
        print(f"epoch: {epoch}, loss: {loss}")
        loss.backward()
        optimizer.step()
    return encoder, decoder

# gmae_encode, gmae_decode, mask_embedding, remask_embedding = make_gmae()
# node_embeddings = make_embeddings(dataset)
# train_gmae(gmae_encode, gmae_decode, mask_embedding, remask_embedding, node_embeddings, data)
# torch.save(gmae_encoder.state_dict(), "./gmae_encoder")
