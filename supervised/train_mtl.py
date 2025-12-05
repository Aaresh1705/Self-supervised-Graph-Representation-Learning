import sys, os
import copy

import torch

from torch_geometric.nn import to_hetero
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.loader import LinkNeighborLoader

from tqdm import tqdm

# We had a problem importing the lib package when a file was inside a folder
# This fixed it
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from lib.dataset import load_data, to_inductive
from lib.model import GraphSAGE, SupervisedMTL


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    root_path = './'
    transform = Compose([ToUndirected(merge=False)])
    preprocess = 'metapath2vec'
    data = load_data(root_path, transform=transform, preprocess=preprocess)

    target_node_type = "paper"
    target_edge_type = ('paper', 'has_topic', 'field_of_study')
    data_inductive = to_inductive(copy.deepcopy(data), target_node_type)

    hidden_dim = 128
    num_classes = int(data[target_node_type].y.max()) + 1

    train_loader = LinkNeighborLoader(
        data_inductive,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type,  data_inductive[target_edge_type].edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=True
    )

    edge_index_all = data[target_edge_type].edge_index
    src_papers = edge_index_all[0]

    # Use papers val_mask to select validation edges
    val_edge_mask = data['paper'].val_mask[src_papers]
    val_edge_index = edge_index_all[:, val_edge_mask]

    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=[15, 10],
        edge_label_index=(target_edge_type, val_edge_index),
        neg_sampling_ratio=1.0,
        batch_size=2048,
        shuffle=False
    )

    model = GraphSAGE(in_channels=hidden_dim, out_channels=hidden_dim, num_classes=num_classes)
    model = to_hetero(model, data_inductive.metadata(), aggr='sum')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    pipeline = SupervisedMTL(
        model=model,
        device=device,
        optimizer=optimizer,
        target_node_type=target_node_type,
        target_edge_type=target_edge_type
    )

    best_loss = float("inf")
    save_path = 'supervised/models/mtl/'
    os.makedirs(save_path, exist_ok=True)

    max_steps = 10_000
    eval_every = 500
    save_every = 500

    step = 0
    epoch = 0

    pbar = tqdm(total=max_steps)
    while step < max_steps:
        print(f"=== Epoch {epoch} ===")
        for batch in train_loader:
            step += 1

            pipeline.train_on_batch(batch)

            if step % eval_every == 0:
                metrics = pipeline.test(val_loader)
                loss = metrics["loss_total"]

                print(
                    f"AUC: {metrics['AUC']:.4f} | "
                    f"Accuracy: {metrics['Accuracy']:.4f} | "
                    f"Edge loss: {metrics['loss_edge']:.4f} | "
                    f"Node loss: {metrics['loss_node']:.4f} | "
                    f"Total Loss: {loss:.4f}"
                )

                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), save_path + "best.pt")

            if step % save_every == 0:
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                }, save_path + f"checkpoint_{step}.pt")

            if step >= max_steps:
                break

            pbar.update(1)
        epoch += 1
