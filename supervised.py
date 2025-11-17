#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain import get_x_dict, gae_encoder, gmae_encoder, code_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_paper_embeddings(encoder, embeddings, data, device):
    encoder.eval()
    data = data.to(device)

    with torch.no_grad():
        x_dict = get_x_dict(data, embeddings)   # your helper
        z_dict = encoder(x_dict, data.edge_index_dict)
        z_paper = z_dict["paper"]               # [num_paper, code_size]

    return z_paper

def train_paper_classifier(encoder, embeddings, data, code_size,
                           num_classes=349, epochs=100, lr=0.01, weight_decay=5e-4):
    data = data.to(device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in embeddings.parameters():
        p.requires_grad = False

    paper_z = get_paper_embeddings(encoder, embeddings, data, device)
    readout = Readout(code_size, num_classes).to(device)
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr, weight_decay=weight_decay)
    paper_y = data["paper"].y.to(device).view(-1)
    train_mask = data["paper"].train_mask.to(device)
    val_mask   = data["paper"].val_mask.to(device)
    test_mask  = data["paper"].test_mask.to(device)

    for epoch in range(epochs):
        readout.train()
        optimizer.zero_grad()

        logits = readout(paper_z)
        loss = F.cross_entropy(logits[train_mask], paper_y[train_mask])
        loss.backward()
        optimizer.step()

        readout.eval()
        with torch.no_grad():
            logits = readout(paper_z)
            pred = logits.argmax(dim=-1)

            def acc(mask):
                if mask.sum() == 0:
                    return 0.0
                return (pred[mask] == paper_y[mask]).float().mean().item()
            train_acc = acc(train_mask)
            val_acc   = acc(val_mask)
            test_acc  = acc(test_mask)
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}"
        )
    return readout

gae_encoder.load_state_dict(torch.load("./gae_encoder", weights_only=True))
gae_encoder.eval()

paper_clf = train_paper_classifier(
    encoder=gae_encoder,   
    embeddings=node_embeddings,
    data=dataset,           
    code_size=code_size,
    num_classes=num_paper_classes,
    epochs=100,)
