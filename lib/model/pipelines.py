import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

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

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            node_count = batch[self.target_type].batch_size
            node_idx = torch.arange(node_count, device=self.device)

            z = h_dict[self.target_type]
            logits = self.model.node_classifier(z, node_idx)

            y = batch[self.target_type].y[node_idx]

            loss = F.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach())

        return total_loss / len(loader)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()

        total_correct = total_examples = 0
        for batch in tqdm(loader, desc='Validating'):
            batch = batch.to(self.device)

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            node_count = batch[self.target_type].batch_size
            node_idx = torch.arange(node_count, device=self.device)

            z = h_dict[self.target_type]
            logits = self.model.node_classifier(z, node_idx)
            pred = logits.argmax(dim=-1)

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

    def train_on_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        edge_label_index = batch[self.target_edge_type].edge_label_index
        edge_labels = batch[self.target_edge_type].edge_label.float().to(self.device)

        # Remove target edges from message passing
        remove_mask = torch.isin(batch.edge_index_dict[self.target_edge_type], edge_label_index).all(dim=0).to(bool)
        eid = batch[self.target_edge_type].edge_index
        batch[self.target_edge_type].edge_index = eid[:, ~remove_mask]

        h_dict = self.model(batch.x_dict, batch.edge_index_dict)

        z_src = h_dict[self.src_type][edge_label_index[0]]
        z_dst = h_dict[self.dst_type][edge_label_index[1]]

        logits = self.model.edge_classifier(z_src, z_dst)
        loss = F.binary_cross_entropy_with_logits(logits, edge_labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(self, loader, max_batches=None):
        self.model.eval()

        preds = []
        ground_truths = []

        pbar = tqdm(loader, desc="Validating")
        batches = 0
        for batch in pbar:
            batches += 1
            batch = batch.to(self.device)

            edge_label_index = batch[self.target_edge_type].edge_label_index

            # Remove target edges from message passing
            remove_mask = torch.isin(batch.edge_index_dict[self.target_edge_type], edge_label_index).all(dim=0).to(bool)
            eid = batch[self.target_edge_type].edge_index
            batch[self.target_edge_type].edge_index = eid[:, ~remove_mask]

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            z_src = h_dict[self.src_type][edge_label_index[0]]
            z_dst = h_dict[self.dst_type][edge_label_index[1]]

            pred = self.model.edge_classifier(z_src, z_dst)
            preds.append(pred)

            ground_truths.append(batch[self.target_edge_type].edge_label)
            if (not max_batches is None) and batches > max_batches:
                break

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)

        return {
            "AUC": auc,
            "acc": ((pred > .5) == ground_truth) / len(preds)
        }


class SupervisedMTL:
    def __init__(self, model, device, optimizer, target_node_type, target_edge_type):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_edge_type = target_edge_type
        self.target_node_type = target_node_type

        self.src_type, _, self.dst_type = target_edge_type

    def train_on_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        edge_label_index = batch[self.target_edge_type].edge_label_index

        remove_mask = torch.isin(batch.edge_index_dict[self.target_edge_type], edge_label_index).all(dim=0).to(bool)
        eid = batch[self.target_edge_type].edge_index
        batch[self.target_edge_type].edge_index = eid[:, ~remove_mask]

        h_dict = self.model(batch.x_dict, batch.edge_index_dict)

        # Edge task
        z_src = h_dict[self.src_type][edge_label_index[0]]
        z_dst = h_dict[self.dst_type][edge_label_index[1]]

        edge_logits = self.model.edge_classifier(z_src, z_dst)
        edge_labels = batch[self.target_edge_type].edge_label.float().to(self.device)
        loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)

        # Node task
        node_idx = torch.unique(edge_label_index[0])

        z = h_dict[self.target_node_type]
        node_logits = self.model.node_classifier(z, node_idx)

        y = batch[self.target_node_type].y[node_idx]
        loss_node = F.cross_entropy(node_logits, y)

        # Combine loss
        total_loss = loss_node + loss_edge
        total_loss.backward()

        self.optimizer.step()

        return {
            "node_loss": loss_node.item(),
            "edge_loss": loss_edge.item(),
            "total_loss": total_loss.item(),
        }

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()

        preds = []
        ground_truths = []
        total_correct = total_examples = 0
        loss_dict = {'edge': [], 'node': [], 'total': []}

        pbar = tqdm(loader, desc="Validating")
        for batch in pbar:
            batch = batch.to(self.device)

            edge_label_index = batch[self.target_edge_type].edge_label_index

            # Remove target edges from message passing
            remove_mask = torch.isin(batch.edge_index_dict[self.target_edge_type], edge_label_index).all(dim=0).to(bool)
            eid = batch[self.target_edge_type].edge_index
            batch[self.target_edge_type].edge_index = eid[:, ~remove_mask]

            h_dict = self.model(batch.x_dict, batch.edge_index_dict)

            # Edge task
            z_src = h_dict[self.src_type][edge_label_index[0]]
            z_dst = h_dict[self.dst_type][edge_label_index[1]]

            edge_pred = self.model.edge_classifier(z_src, z_dst)
            preds.append(edge_pred)

            ground_truths.append(batch[self.target_edge_type].edge_label)

            edge_logits = self.model.edge_classifier(z_src, z_dst)
            edge_labels = batch[self.target_edge_type].edge_label.float().to(self.device)
            loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            loss_dict['edge'].append(loss_edge.item())

            # Node task
            node_idx = torch.unique(edge_label_index[0])

            z = h_dict[self.target_node_type]
            node_logits = self.model.node_classifier(z, node_idx)

            node_pred = node_logits.argmax(dim=-1)
            y = batch[self.target_node_type].y[node_idx]

            total_correct += int((node_pred == y).sum())
            total_examples += y.size(0)

            loss_node = F.cross_entropy(node_logits, y)
            loss_dict['node'].append(loss_node.item())

            total_loss = loss_node + loss_edge
            loss_dict['total'].append(total_loss.item())

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)

        return {
            "AUC": auc,
            "Accuracy": total_correct / total_examples,
            "loss_edge": np.mean(loss_dict['edge']),
            "loss_node": np.mean(loss_dict['node']),
            "loss_total": np.mean(loss_dict['total'])
        }


def pretrain_gae(data):
    gae_encoder, gae_decoder = make_gae()
    def train_gae(encoder, decoder, data):
        optimizer = torch.optim.Adam(  list(encoder.parameters())
                                    + list(decoder.parameters()), lr=0.01)
        for epoch in range(500):
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
