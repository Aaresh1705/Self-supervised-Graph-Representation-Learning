from torch_geometric.datasets import OGB_MAG
from torch_geometric.data import HeteroData
import torch


def load_data(root='', preprocess='metapath2vec', transform=None) -> HeteroData:
    data = OGB_MAG(root=root, preprocess=preprocess, transform=transform)[0]

    return data


def to_inductive(data: HeteroData, node_type: str) -> HeteroData:
    """
    A function that removes all val/test node features and edges between train nodes and val/test nodes.

    """
    train_mask = data[node_type].train_mask
    train_mask_idxs = torch.where(train_mask)[0]
    N_train = len(train_mask_idxs)

    # define new edge index
    new_paper_idxs = torch.full((len(train_mask),), -1, device=train_mask.device)
    new_paper_idxs[train_mask] = torch.arange(N_train, device=train_mask.device)

    # restrict node_type to only include train split
    data[node_type].x = data[node_type].x[train_mask]
    data[node_type].y = data[node_type].y[train_mask]
    data[node_type].year = data[node_type].year[train_mask]
    data[node_type].train_mask = torch.ones((N_train), dtype=torch.bool, device=train_mask.device)
    data[node_type].val_mask = torch.zeros((N_train), dtype=torch.bool, device=train_mask.device)
    data[node_type].test_mask = torch.zeros((N_train), dtype=torch.bool, device=train_mask.device)

    # find edges with node_type as either source or destination
    edge_types = list(data.edge_index_dict.keys())
    edge_type_mask = [(e[0] == node_type, e[-1] == node_type) for e in edge_types]

    edge_index_dict = data.edge_index_dict

    for i, edge_type in enumerate(edge_types):
        if not any(edge_type_mask[i]):
            continue

        edge_index = edge_index_dict[edge_type]
        src_mask = torch.ones((edge_index.size(1)), dtype=bool)
        dst_mask = torch.ones((edge_index.size(1)), dtype=bool)

        # mask paper nodes in edge index not part of train
        if edge_type[0] == node_type:
            src_mask = new_paper_idxs[edge_index[0]] != -1

        if edge_type[-1] == node_type:
            dst_mask = new_paper_idxs[edge_index[1]] != -1

        edge_mask = src_mask & dst_mask
        filtered_edge_index = edge_index[:, edge_mask]

        if edge_type[0] == node_type:
            filtered_edge_index[0] = new_paper_idxs[filtered_edge_index[0]]

        if edge_type[-1] == node_type:
            filtered_edge_index[1] = new_paper_idxs[filtered_edge_index[1]]

        data[edge_type]['edge_index'] = filtered_edge_index

    return data


# def verify_inductive(data: HeteroData, target) -> None:
#     edge_index = data['paper', 'cites', 'paper'].edge_index
#     src, dst = edge_index
#
#     train_mask = data['paper'].train_mask
#     val_mask = data['paper'].val_mask
#     test_mask = data['paper'].test_mask
#
#     # Check if any paper-paper edge touches val/test nodes
#     bad_src = (~train_mask[src]).sum().item()
#     bad_dst = (~train_mask[dst]).sum().item()
#
#     print(f"Edges touching val/test papers: src={bad_src}, dst={bad_dst}")
#
#
#     print('Train mask:')
#     mask = data[target].train_mask  # example: a boolean tensor
#     num_true = mask.sum().item()  # count True values
#     num_false = (~mask).sum().item()  # count False value
#     print(f"True: {num_true}, False: {num_false}")
#
#     print('Validation mask:')
#     mask = data[target].val_mask  # example: a boolean tensor
#     num_true = mask.sum().item()  # count True values
#     num_false = (~mask).sum().item()  # count False value
#     print(f"True: {num_true}, False: {num_false}")
#
#     print('Test mask:')
#     mask = data[target].test_mask  # example: a boolean tensor
#     num_true = mask.sum().item()  # count True values
#     num_false = (~mask).sum().item()  # count False value
#     print(f"True: {num_true}, False: {num_false}")