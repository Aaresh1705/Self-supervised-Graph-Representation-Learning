import sys, os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

# We had a problem importing the lib package when a file was inside a folder
# This fixed it
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from lib.dataset import load_data
from lib.model import SupervisedMTL, GraphSAGE


@torch.no_grad()
def confusion_matrix_for_paper(model, data, paper_id, target_edge_type, device):
    model.eval()
    model = model.to(device)
    data = data.to(device)

    src_type, _, dst_type = target_edge_type

    h_dict = model(data.x_dict, data.edge_index_dict)

    paper_emb = h_dict[src_type][paper_id]  # [D]
    field_emb = h_dict[dst_type]  # [num_fields, D]

    num_fields = field_emb.size(0)

    # Predict probabilities for all fields
    paper_rep = paper_emb.unsqueeze(0).repeat(num_fields, 1)
    logits = model.edge_classifier(paper_rep, field_emb)
    probs = torch.sigmoid(logits)

    pred_mask = (probs > 0.5).cpu().numpy()  # predicted positives

    # Get true labels for all fields
    edge_index = data[target_edge_type].edge_index.cpu()
    true_mask = (edge_index[0] == paper_id)

    true_fields = edge_index[1][true_mask].numpy()

    true_mask_full = np.zeros(num_fields, dtype=bool)
    true_mask_full[true_fields] = True  # mark true edges

    # Confusion matrix components
    TP = np.sum(pred_mask & true_mask_full)
    FP = np.sum(pred_mask & ~true_mask_full)
    FN = np.sum(~pred_mask & true_mask_full)
    TN = np.sum(~pred_mask & ~true_mask_full)

    print("\n=== Confusion Matrix for Paper", paper_id, "===")
    print(f"TP (correct predicted edges):      {TP}")
    print(f"FP (predicted but not true):       {FP}")
    print(f"FN (missed true edges):            {FN}")
    print(f"TN (correctly ignored fields):     {TN}")
    print("-------------------------------------------")
    print(f"Positive Recall (TP / (TP+FN)):    {TP / max(TP + FN, 1):.4f}")
    print(f"Precision (TP / (TP+FP)):          {TP / max(TP + FP, 1):.4f}")
    print(f"Accuracy:                          {(TP + TN) / (TP + TN + FP + FN):.4f}")
    print("-------------------------------------------")

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "recall": TP / max(TP + FN, 1),
        "precision": TP / max(TP + FP, 1),
        "accuracy": (TP + TN) / (TP + TN + FP + FN)
    }


@torch.no_grad()
def visualize_paper_prediction(model, data, paper_id, target_edge_type, device):
    model.eval()
    model = model.to(device)
    data = data.to(device)

    src_type, _, dst_type = target_edge_type

    # Sample subgraph around the paper
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        input_nodes=("paper", torch.tensor([paper_id])),
        batch_size=1,
        shuffle=False
    )

    batch = next(iter(loader)).to(device)

    # Map global paper id to batch index
    batch_paper_id = (batch[src_type].n_id == paper_id).nonzero(as_tuple=True)[0].item()

    h_dict = model(batch.x_dict, batch.edge_index_dict)

    paper_emb = h_dict[src_type][batch_paper_id]

    # Global field ids present in this batch
    batch_field_global_ids = batch[dst_type].n_id.cpu().numpy()

    field_emb = h_dict[dst_type]
    num_fields = field_emb.size(0)

    # Compute scores for paper to every field in batch
    paper_rep = paper_emb.unsqueeze(0).repeat(num_fields, 1)
    logits = model.edge_classifier(paper_rep, field_emb)
    probs = torch.sigmoid(logits).cpu().numpy()

    # Predictions: convert local batch indices to global field ids
    predicted_local = np.where(probs > 0.5)[0]
    predicted_global = batch_field_global_ids[predicted_local]

    # Get the true field edges for the original graph
    full_edge_index = data[target_edge_type].edge_index.cpu()

    true_mask_global = (full_edge_index[0] == paper_id)
    true_fields_global = full_edge_index[1][true_mask_global].cpu().numpy()

    G = nx.Graph()
    G.add_node(f"paper_{paper_id}", color="black", size=800)

    # Nodes that appear either in truth or prediction
    fields_to_display = set(true_fields_global.tolist()) | set(predicted_global.tolist())

    for f in fields_to_display:
        G.add_node(f"field_{f}", color="lightgray", size=300)

        is_true = f in true_fields_global
        is_pred = f in predicted_global

        if is_true and is_pred:
            edge_color = "green"  # true positive
        elif is_true and not is_pred:
            edge_color = "red"  # false negative
        elif is_pred and not is_true:
            edge_color = "blue"  # false positive
        else:
            continue

        G.add_edge(f"paper_{paper_id}", f"field_{f}", color=edge_color)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[G.nodes[n]["color"] for n in G.nodes()],
        node_size=[G.nodes[n]["size"] for n in G.nodes()],
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[G[u][v]["color"] for u, v in G.edges()],
        width=2
    )
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title(f"Predicted Fields for Paper {paper_id}\n(green=TP, red=FN, blue=FP)")
    plt.axis("off")

    os.makedirs("plots", exist_ok=True)
    plt.savefig("supervised/plots/paper_graph_prediction_mtl.png", dpi=300)
    plt.close()

    print("Saved visualization to supervised/plots/paper_graph_prediction_mtl.png")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    root_path = './'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = load_data(root_path, transform=transform, preprocess=preprocess)

    target_node_type = "paper"
    target_edge_type = ('paper', 'has_topic', 'field_of_study')


    edge_index_all = data[target_edge_type].edge_index
    src_papers = edge_index_all[0]

    # Use papers test_mask to select test edges:
    test_edge_mask = data['paper'].test_mask[src_papers]
    test_edge_index = edge_index_all[:, test_edge_mask]

    test_loader = LinkNeighborLoader(
        data,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, test_edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=False,
    )

    hidden_dim = 128
    num_classes = int(data[target_node_type].y.max()) + 1

    model = GraphSAGE(in_channels=hidden_dim, out_channels=hidden_dim, num_classes=num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')
    model = model.to(device)

    best_model_path = "supervised/models/mtl/best.pt"
    print(f"Loading best model from: {best_model_path}")

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    pipeline = SupervisedMTL(
        model=model,
        device=device,
        optimizer=None,
        target_node_type=target_node_type,
        target_edge_type=target_edge_type
    )

    test_metrics = pipeline.test(test_loader)
    print(f"\n=== Test AUC: {test_metrics['AUC']:.4f} Accuracy: {test_metrics['Accuracy']:.4f} ===\n")

    paper_id = torch.where(test_edge_mask)[0][0].item()
    visualize_paper_prediction(
        model=model,
        data=data,
        paper_id=paper_id,
        target_edge_type=('paper', 'has_topic', 'field_of_study'),
        device=device
    )
    confusion_matrix_for_paper(
        model=model,
        data=data,
        paper_id=3484,
        target_edge_type=('paper', 'has_topic', 'field_of_study'),
        device=device
    )