#!/usr/bin/env python3

# moved to pipelines.py

import torch

from model import make_gae, get_x_dict, make_embeddings
from dataset import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_data(remove_test=True)
gae_encoder, gae_decoder = make_gae()
node_embeddings = make_embeddings(data)

def train_gae(encoder, decoder, embeddings, data):
    optimizer = torch.optim.Adam(  list(encoder.parameters())
                                 + list(decoder.parameters())
                                 + list(embeddings.parameters()), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        x_dict = get_x_dict(data, embeddings)
        z = encoder(x_dict, data.edge_index_dict)
        x = decoder(z["paper"])
        loss = torch.nn.functional.mse_loss(x, data["paper"].x)
        print(f"epoch: {epoch}, loss: {loss}")
        loss.backward()
        optimizer.step()
    return encoder, decoder

# train_gae(gae_encoder, gae_decoder, node_embeddings, dataset)
# torch.save(gae_encoder.state_dict(), "./gae_encoder")
