from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

from lib.model.models import GraphSAGE

from . import models
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SupervisedNodePredictions:
    def __init__(self, model: GraphSAGE, device, optimizer, target_type):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_type = target_type

    def train(self, loader):
        self.model.train()

        total_loss = 0.0
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)  # <-- run on the BATCH

            seed_count = batch[self.target_type].batch_size
            seed_idx = torch.arange(seed_count, device=self.device)

            Z = h_dict[self.target_type]
            logits = self.model.node_classifier(Z, seed_idx)

            y = batch[self.target_type].y[seed_idx]

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

            node_idx = torch.arange(batch[self.target_type].batch_size, device=self.device)
            logits = self.model.node_classifier(Z, node_idx)
            pred = logits.argmax(dim=-1)  # ✅ Correct

            # Labels
            y = batch[self.target_type].y[:batch[self.target_type].batch_size].view(-1)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)
        return total_correct / total_examples


class SupervisedEdgePredictions:
    def __init__(self, model: GraphSAGE, device, optimizer, target_edge_type):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_edge_type = target_edge_type

        self.src_type = target_edge_type[0]
        self.dst_type = target_edge_type[2]

    def train(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)  # <-- run on the BATCH

            src_emb = h_dict[self.src_type]
            dst_emb = h_dict[self.dst_type]

            edge_label_index = batch[self.target_edge_type].edge_label_index
            edge_labels = batch[self.target_edge_type].edge_label.float().to(self.device)

            src_indices = edge_label_index[0]
            dst_indices = edge_label_index[1]

            src_edge_emb = src_emb[src_indices]
            dst_edge_emb = dst_emb[dst_indices]

            logits = self.model.edge_classifier(src_edge_emb, dst_edge_emb)
            loss = F.binary_cross_entropy_with_logits(logits, edge_labels)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            pos_mask = edge_labels == 1
            batch_pos_acc = (preds[pos_mask] == 1).float().mean()

            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_pos_acc:.4f}",
            })

            total_loss += float(loss.detach())

        return total_loss / len(loader)  # <-- use loader

    @torch.no_grad()
    def test(self, loader, k=100):
        self.model.eval()

        all_probs = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            src_emb = h_dict[self.src_type]
            dst_emb = h_dict[self.dst_type]

            edge_label_index = batch[self.target_edge_type].edge_label_index
            edge_labels = batch[self.target_edge_type].edge_label.float().to(self.device)

            src_idx = edge_label_index[0]
            dst_idx = edge_label_index[1]

            src_edge_emb = src_emb[src_idx]
            dst_edge_emb = dst_emb[dst_idx]

            logits = self.model.edge_classifier(src_edge_emb, dst_edge_emb)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(edge_labels.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # ----------------------------------
        # 1. ACCURACY (basic)
        # ----------------------------------
        preds = (all_probs > 0.5).astype(int)
        accuracy = (preds == all_labels).mean()

        # ----------------------------------
        # 2. BALANCED ACCURACY
        # ----------------------------------
        try:
            bacc = balanced_accuracy_score(all_labels, preds)
        except:
            bacc = 0.0

        # ----------------------------------
        # 3. F1 SCORE
        # ----------------------------------
        try:
            f1 = f1_score(all_labels, preds)
        except:
            f1 = 0.0

        # ----------------------------------
        # 4. AUROC
        # ----------------------------------
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.0

        # ----------------------------------
        # 5. AUPRC
        # ----------------------------------
        try:
            auprc = average_precision_score(all_labels, all_probs)
        except:
            auprc = 0.0

        # ----------------------------------
        # 6. PRECISION@K
        # ----------------------------------
        topk_idx = np.argsort(-all_probs)[:k]
        precision_at_k = all_labels[topk_idx].mean()

        # ----------------------------------
        # 7. RECALL@K
        # ----------------------------------
        total_positives = all_labels.sum()
        recall_at_k = all_labels[topk_idx].sum() / total_positives

        return {
            "accuracy": accuracy,
            "balanced_accuracy": bacc,
            "f1": f1,
            "auroc": auroc,
            "auprc": auprc,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
        }


class SupervisedMTL:
    def __init__(self, model, device, optimizer, target_node_type, target_edge_type):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_edge_type = target_edge_type
        self.target_node_type = target_node_type

        self.src_type, _, self.dst_type = target_edge_type

    def train(self, loader):
        self.model.train()

        total_node_loss = 0.0
        total_edge_loss = 0.0
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            n_batches += 1
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # --------------------------------------------------------
            # 1) Encode heterogeneous batch
            # --------------------------------------------------------
            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            # --------------------------------------------------------
            # 2) Edge task
            # --------------------------------------------------------
            edge_index = batch[self.target_edge_type].edge_label_index
            edge_labels = batch[self.target_edge_type].edge_label.float()

            src_emb = h_dict[self.src_type][edge_index[0]]
            dst_emb = h_dict[self.dst_type][edge_index[1]]

            edge_logits = self.model.edge_classifier(src_emb, dst_emb)
            loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)

            # Running edge accuracy
            edge_probs = torch.sigmoid(edge_logits)
            edge_preds = (edge_probs > 0.5).float()
            pos_mask = edge_labels == 1
            batch_edge_pos_acc = (edge_preds[pos_mask] == 1).float().mean()

            # --------------------------------------------------------
            # 3) Node task
            # --------------------------------------------------------
            Z = h_dict[self.target_node_type]

            # Node indices to supervise
            if self.src_type == self.target_node_type:
                node_idx = torch.unique(edge_index[0])
            else:
                node_idx = torch.arange(Z.size(0), device=self.device)

            node_logits = self.model.node_classifier(Z, node_idx)
            node_labels = batch[self.target_node_type].y[node_idx]

            loss_node = F.cross_entropy(node_logits, node_labels)

            # Running node accuracy
            node_preds = node_logits.argmax(dim=-1)
            batch_node_acc = float((node_preds == node_labels).float().mean())

            # --------------------------------------------------------
            # 4) Combined Loss (Uncertainty-weighted)
            # --------------------------------------------------------
            batch_loss = self.model.weighted_loss(loss_node, loss_edge)
            batch_loss.backward()
            self.optimizer.step()

            total_node_loss += loss_node.detach().item()
            total_edge_loss += loss_edge.detach().item()
            total_loss += batch_loss.detach().item()

            pbar.set_postfix({
                "node_acc": f"{batch_node_acc:.4f}",
                "edge_acc": f"{batch_edge_pos_acc:.4f}",
                "loss": f"{batch_loss:.4f}",
                "node_loss": f"{loss_node:.4f}",
                "edge_loss": f"{loss_edge:.4f}",
                "(sigma_node|sigma_edge)": f"{self.model.weighted_loss.log_sigma_node.detach().item():.4f}|"
                                           f"{self.model.weighted_loss.log_sigma_edge.detach().item():.4f}",
            })

        return {
            "node_loss": total_node_loss / n_batches,
            "edge_loss": total_edge_loss / n_batches,
            "total_loss": total_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()

        total_node_loss = 0.0
        total_edge_loss = 0.0
        n_node_batches = 0
        n_edge_batches = 0

        all_node_preds = []
        all_node_labels = []
        all_edge_probs = []
        all_edge_labels = []

        for batch in loader:
            batch = batch.to(self.device)

            # Shared encoder
            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            # ----------------------------------------------------
            # NODE TASK
            # ----------------------------------------------------
            Z = h_dict[self.target_node_type]
            node_labels_full = batch[self.target_node_type].y

            mask = batch[self.target_node_type].val_mask
            node_idx = mask.nonzero(as_tuple=True)[0]

            if len(node_idx) > 0:
                node_logits = self.model.node_classifier(Z, node_idx)
                node_labels = node_labels_full[node_idx]

                loss_node = F.cross_entropy(node_logits, node_labels)
                total_node_loss += loss_node.item()
                n_node_batches += 1

                node_preds = node_logits.argmax(dim=-1)
                all_node_preds.append(node_preds.cpu())
                all_node_labels.append(node_labels.cpu())

            # ----------------------------------------------------
            # EDGE TASK
            # ----------------------------------------------------
            edge_index = batch[self.target_edge_type].edge_label_index
            edge_labels = batch[self.target_edge_type].edge_label.float()

            src_emb = h_dict[self.src_type][edge_index[0]]
            dst_emb = h_dict[self.dst_type][edge_index[1]]

            edge_logits = self.model.edge_classifier(src_emb, dst_emb)
            loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)

            total_edge_loss += loss_edge.item()
            n_edge_batches += 1

            probs = torch.sigmoid(edge_logits)
            all_edge_probs.append(probs.cpu())
            all_edge_labels.append(edge_labels.cpu())

        # ==========================
        # GLOBAL NODE METRICS
        # ==========================
        if len(all_node_preds) > 0:
            all_node_preds = torch.cat(all_node_preds)
            all_node_labels = torch.cat(all_node_labels)

            node_acc = (all_node_preds == all_node_labels).float().mean().item()
            node_f1 = f1_score(all_node_labels, all_node_preds, average="micro")
        else:
            node_acc = 0.0
            node_f1 = 0.0

        # ==========================
        # GLOBAL EDGE METRICS
        # ==========================
        all_edge_probs = torch.cat(all_edge_probs).numpy()
        all_edge_labels = torch.cat(all_edge_labels).numpy()

        edge_acc = ((all_edge_probs > 0.5).astype(int) == all_edge_labels).mean()

        try:
            edge_auroc = roc_auc_score(all_edge_labels, all_edge_probs)
        except:
            edge_auroc = 0.0

        try:
            edge_auprc = average_precision_score(all_edge_labels, all_edge_probs)
        except:
            edge_auprc = 0.0

        # ==========================
        # COMPUTE UNCERTAINTY-WEIGHTED VAL LOSS
        # ==========================
        avg_node_loss = total_node_loss / max(1, n_node_batches)
        avg_edge_loss = total_edge_loss / max(1, n_edge_batches)

        weighted_val_loss = (
            self.model.weighted_loss(
                torch.tensor(avg_node_loss, device=self.device),
                torch.tensor(avg_edge_loss, device=self.device)
            ).item()
        )

        return {
            "node_loss": avg_node_loss,
            "edge_loss": avg_edge_loss,
            "weighted_loss": weighted_val_loss,
            "node_acc": node_acc,
            "node_f1": node_f1,
            "edge_acc": edge_acc,
            "edge_auroc": edge_auroc,
            "edge_auprc": edge_auprc,
        }


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
    gamma = 1.5
    gmae_mask_rate = .5
    data = data.to(device)

    def sce_loss(true, pred, mask, eps=1e-8): # need mask for determing which nodes were trained on
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
        for epoch in range(300):
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
            z_dict["paper"] = z_paper_masked
            pred = decoder(z_dict, data.edge_index_dict)
            loss = sce_loss(paper_x, pred["paper"], mask)
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
    for z, y in tqdm(loader, "training..."):
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
