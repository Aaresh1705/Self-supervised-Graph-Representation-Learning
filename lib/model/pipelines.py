from tqdm import tqdm
import torch.nn.functional as F
import torch

from . import models
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_supervised(model, device, optimizer, loader, feature_encoder, target_type):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        x_dict = feature_encoder(batch.x_dict)               # <-- encode the BATCH
        out_dict = model(x_dict, batch.edge_index_dict)      # <-- run on the BATCH
        out = out_dict[target_type][:batch[target_type].batch_size]
        y = batch[target_type].y[:batch[target_type].batch_size].view(-1)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / len(loader)                          # <-- use loader

@torch.no_grad()
def test(model, device, loader, feature_encoder, target_type):
    model.eval()
    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        x_dict = feature_encoder(batch.x_dict)               # <-- encode the BATCH
        out_dict = model(x_dict, batch.edge_index_dict)      # <-- run on the BATCH
        out = out_dict[target_type][:batch[target_type].batch_size]
        y = batch[target_type].y[:batch[target_type].batch_size].view(-1)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == y).sum())
        total_examples += y.size(0)
    return total_correct / total_examples

def pretrain_gae(data):
    gae_encoder, gae_decoder = make_gae()
    node_embeddings = make_embeddings(data)
    def train_gae(encoder, decoder, embeddings, data):
        optimizer = torch.optim.Adam(  list(encoder.parameters())
                                    + list(decoder.parameters())
                                    + list(embeddings.parameters()), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            x_dict = models.get_x_dict(data, embeddings)
            z = encoder(x_dict, data.edge_index_dict)
            x = decoder(z["paper"])
            loss = torch.nn.functional.mse_loss(x, data["paper"].x)
            print(f"epoch: {epoch}, loss: {loss}")
            loss.backward()
            optimizer.step()
        return encoder, decoder
    encoder_decoder = train_gae(gae_encoder, gae_decoder, node_embeddings, dataset)
    torch.save(encoder.state_dict(), "./gae_encoder")

def pretrain_gmae(data):
    code_size = 16
    gamma = 1
    gmae_mask_rate = .5

    def sce_loss(true, pred, mask, eps=1e-8):
        true = true[mask]
        pred = pred[mask]
        true_n = true / (true.norm(dim=-1, keepdim=True) + eps)
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + eps)
        loss = (1.0 - (true_n * pred_n).sum(dim=-1)).clamp(min=0) ** gamma
        return loss.mean()

    def train_gmae(encoder, decoder, mask_embedding, remask_embedding, node_embeddings, data):
        optimizer = torch.optim.Adam( list(encoder.parameters())
                                    + list(decoder.parameters())
                                    + list(node_embeddings.parameters())
                                    + [mask_embedding, remask_embedding],
                                    lr=0.01)
        data = data.to(device)
        paper_x = data["paper"].x.to(device)
        num_paper = data["paper"].num_nodes
        for epoch in range(10):
            encoder.train()
            decoder.train()
            optimizer.zero_grad()
            data = data.to(device)
            x_dict = models.get_x_dict(data, node_embeddings)
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

    gmae_encoder, gmae_decoder, mask_embedding, remask_embedding = models.make_gmae()
    node_embeddings = models.make_embeddings(data)
    encoder, _ = train_gmae(gmae_encoder, gmae_decoder, mask_embedding, remask_embedding, node_embeddings, data)
    torch.save(gmae_encoder.state_dict(), "./gmae_encoder")
    torch.save(node_embeddings.state_dict(), "./gmae_encoder_node_embeddings")
