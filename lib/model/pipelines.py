from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from . import models
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SupervisedNodePredictions:
    def __init__(self, model, device, optimizer, target_type):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_type = target_type
        self.classifier = self.Classifier(model.output_dim, model.num_classes).to(self.device)

    class Classifier(nn.Module):
        def __init__(self, output_dim, num_classes):
            super().__init__()
            self.head = nn.Linear(output_dim, num_classes)
        def forward(self, Z, node_idx):
            Z_batch = Z[node_idx]           # select the relevant nodes
            logits = self.head(Z_batch)     # shape [batch_size, num_classes]
            return logits

    def train(self, loader):
        self.model.train()
        self.classifier.train()

        total_loss = 0.0
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)  # <-- run on the BATCH
            Z = h_dict[self.target_type]  # node embeddings for this type

            # Which nodes in this batch?
            node_idx = torch.arange(
                batch[self.target_type].batch_size,
                device=device
            )

            # Apply classifier on just the batch nodes
            logits = self.classifier(Z, node_idx)

            # Labels
            y = batch[self.target_type].y[:batch[self.target_type].batch_size].view(-1)

            # Compute loss
            loss = F.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach())
        return total_loss / len(loader)  # <-- use loader

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = total_examples = 0
        for batch in tqdm(loader, desc='Validating'):
            batch = batch.to(self.device)
            x_dict = batch.x_dict
            h_dict = self.model(x_dict, batch.edge_index_dict)  # <-- run on the BATCH
            Z = h_dict[self.target_type]  # node embeddings for this type

            node_idx = torch.arange(batch[self.target_type].batch_size, device=device)
            logits = self.classifier(Z, node_idx)
            pred = logits.argmax(dim=-1)  # ✅ Correct

            # Labels
            y = batch[self.target_type].y[:batch[self.target_type].batch_size].view(-1)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)
        return total_correct / total_examples


class SupervisedEdgePredictions:
    def __init__(self, model, device, optimizer, target_edge_type):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_edge_type = target_edge_type

        self.src_type = target_edge_type[0]
        self.dst_type = target_edge_type[2]

        self.edge_decoder = self.DotProductDecoder()

    class DotProductDecoder(nn.Module):
        """
        Simpler decoder that uses dot product between embeddings.
        Good for symmetric relationships.
        """
        def __init__(self):
            super().__init__()

        def forward(self, src_emb, dst_emb):
            logits = (src_emb * dst_emb).sum(dim=-1)
            return logits

    def train(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)  # <-- run on the BATCH

            src_emb = h_dict[self.src_type]
            dst_emb = h_dict[self.dst_type]

            edge_label_index = batch[self.target_edge_type].edge_label_index
            edge_labels = batch[self.target_edge_type].edge_label

            src_indices = edge_label_index[0]
            dst_indices = edge_label_index[1]

            src_edge_emb = src_emb[src_indices]
            dst_edge_emb = dst_emb[dst_indices]

            logits = self.edge_decoder(src_edge_emb, dst_edge_emb)
            loss = F.binary_cross_entropy_with_logits(logits, edge_labels)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach())

        return total_loss / len(loader)  # <-- use loader

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()

        all_probs = []
        all_labels= []

        total_correct = 0
        total_examples = 0

        for batch in tqdm(loader, desc='Validating'):
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)  # <-- run on the BATCH

            src_emb = h_dict[self.src_type]
            dst_emb = h_dict[self.dst_type]

            edge_label_index = batch[self.target_edge_type].edge_label_index
            edge_labels = batch[self.target_edge_type].edge_label

            src_indices = edge_label_index[0]
            dst_indices = edge_label_index[1]

            src_edge_emb = src_emb[src_indices]
            dst_edge_emb = dst_emb[dst_indices]

            logits = self.edge_decoder(src_edge_emb, dst_edge_emb)
            probs = F.sigmoid(logits)

            preds = (probs > 0.5).long()
            total_correct += int((preds == edge_labels).sum())
            total_examples += edge_labels.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(edge_labels.cpu())

        accuracy = total_correct / total_examples

        # Compute AUROC and AUPRC
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            auroc = roc_auc_score(all_labels, all_probs)
            auprc = average_precision_score(all_labels, all_probs)
        except:
            auroc = 0.0
            auprc = 0.0

        return accuracy, auroc, auprc


def pretrain_gae(data):
    gae_encoder, gae_decoder = make_gae()
    def train_gae(encoder, decoder, data):
        optimizer = torch.optim.Adam(  list(encoder.parameters())
                                    + list(decoder.parameters()), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            x_dict = models.get_x_dict(data)
            z = encoder(x_dict, data.edge_index_dict)
            x = decoder(z["paper"])
            loss = torch.nn.functional.mse_loss(x, data["paper"].x)
            print(f"epoch: {epoch}, loss: {loss}")
            loss.backward()
            optimizer.step()
        return encoder, decoder
    encoder, decoder = train_gae(gae_encoder, gae_decoder, dataset)
    return encoder

def pretrain_gmae(data):
    code_size = 16
    gamma = 1
    gmae_mask_rate = .5
    data = data.to(device)

    def sce_loss(true, pred, mask, eps=1e-8):
        true = true[mask]
        pred = pred[mask]
        true_n = true / (true.norm(dim=-1, keepdim=True) + eps)
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + eps)
        loss = (1.0 - (true_n * pred_n).sum(dim=-1)).clamp(min=0) ** gamma
        return loss.mean()

    def train_gmae(encoder, decoder, mask_embedding, remask_embedding, data):
        optimizer = torch.optim.Adam( list(encoder.parameters())
                                    + list(decoder.parameters())
                                    + [mask_embedding, remask_embedding],
                                    lr=0.01)
        paper_x = data["paper"].x
        num_paper = data["paper"].num_nodes
        for epoch in range(2):
            encoder.train()
            decoder.train()
            optimizer.zero_grad()
            data = data.to(device)
            x_dict = models.get_x_dict(data)
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

    gmae_encoder, gmae_decoder, mask_embedding, remask_embedding = map(lambda p: p.to(device), models.make_gmae())
    train_gmae(gmae_encoder, gmae_decoder, mask_embedding, remask_embedding, data)
    return gmae_encoder

@torch.no_grad()
def test_node_readout(readout_model, loader):
    readout_model.eval()
    n_examples = 0
    n_correct = 0
    for z, y in tqdm(loader, "validating..."):
        z = z.to(device)
        y = y.to(device)
        logits = readout_model(z)
        pred = logits.argmax(dim=-1)
        n_correct += int((pred == y).sum())
        n_examples += y.size(0)
    return n_correct / n_examples

def train_node_readout(readout_model, loader):
    readout_model.train()
    optimizer = torch.optim.Adam(list(readout_model.parameters()), lr=0.003)
    total_loss = 0.0
    for z, y in tqdm(loader, "testing..."):
        z = z.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = readout_model(z)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        total_loss += float(loss.detach())
        optimizer.step()
    return total_loss / len(loader)

def train_edge_readout(readout, loader):
    readout.train()
    optimizer = torch.optim.Adam(readout.parameters(), lr=1e-3)

    total_loss = 0.0
    total_examples = 0

    for z_src, z_dst, y in tqdm(loader, "training..."):
        optimizer.zero_grad()
        h_src = readout(z_src)  
        h_dst = readout(z_dst) 
        logits = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)

@torch.no_grad()
def test_edge_readout(readout, loader):
    readout.eval()
    total = 0
    correct = 0

    for z_src, z_dst, y in tqdm(loader, "validating ..."):
        h_src = readout(z_src)
        h_dst = readout(z_dst)
        score = (h_src * h_dst).sum(dim=-1)
        pred = ((score.sigmoid()) > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)
