from torch_geometric.datasets import OGB_MAG
from torch_geometric.data import HeteroData
import torch

def load_data(root='', preprocess='metapath2vec', transform=None, remove_test=False) -> HeteroData:
    data = OGB_MAG(root=root, preprocess=preprocess, transform=transform)[0]
    if remove_test:
        data = subset_ogbn_mag_to_train_papers(data)
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

metadata = (['paper', 'author', 'institution', 'field_of_study'], [('author', 'affiliated_with', 'institution'), ('author', 'writes', 'paper'), ('paper', 'cites', 'paper'), ('paper', 'has_topic', 'field_of_study'), ('institution', 'rev_affiliated_with', 'author'), ('paper', 'rev_writes', 'author'), ('paper', 'rev_cites', 'paper'), ('field_of_study', 'rev_has_topic', 'paper')])

def subset_ogbn_mag_to_train_papers(data: HeteroData) -> HeteroData:
    """
    Create a subgraph of ogbn-mag that contains only the *train* paper nodes
    (according to data['paper'].train_mask) and all non-paper nodes.
    Edges incident to val/test paper nodes are removed and paper indices are
    relabeled. Other node types are copied as-is.
    """
    new_data = HeteroData()

    # ---- 1) Subset paper nodes ----
    paper = data['paper']
    train_mask = paper.train_mask
    num_paper = paper.x.size(0)
    num_train_paper = int(train_mask.sum())

    # mapping: old paper index -> new paper index (or -1 if dropped)
    paper_old2new = -torch.ones(num_paper, dtype=torch.long)
    paper_old2new[train_mask] = torch.arange(num_train_paper)

    # copy paper node attributes that are node-wise
    for key, value in paper.items():
        # skip original masks; we'll recreate them
        if key in ['train_mask', 'val_mask', 'test_mask']:
            continue

        if torch.is_tensor(value) and value.size(0) == num_paper:
            # node-level attribute (e.g. x, y, something per node)
            new_data['paper'][key] = value[train_mask]
        else:
            # global attributes (e.g. num_nodes, something scalar)
            new_data['paper'][key] = value

    # recreate masks for the new paper node set
    new_data['paper'].train_mask = torch.ones(num_train_paper, dtype=torch.bool)
    new_data['paper'].val_mask   = torch.zeros(num_train_paper, dtype=torch.bool)
    new_data['paper'].test_mask  = torch.zeros(num_train_paper, dtype=torch.bool)

    # ---- 2) Copy other node types unchanged ----
    for node_type in data.node_types:
        if node_type == 'paper':
            continue

        for key, value in data[node_type].items():
            # shallow copy is typically fine; use .clone() if you’ll mutate
            new_data[node_type][key] = value.clone() if torch.is_tensor(value) else value

    # ---- 3) Filter and remap edge types ----
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        store = data[edge_type]

        edge_index = store.edge_index
        src, dst = edge_index

        # start with all edges kept
        edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool)

        # drop edges touching removed paper nodes
        if src_type == 'paper':
            edge_mask &= train_mask[src]
        if dst_type == 'paper':
            edge_mask &= train_mask[dst]

        src = src[edge_mask]
        dst = dst[edge_mask]

        # remap paper indices
        if src_type == 'paper':
            src = paper_old2new[src]
        if dst_type == 'paper':
            dst = paper_old2new[dst]

        new_edge_index = torch.stack([src, dst], dim=0)
        new_data[edge_type].edge_index = new_edge_index

        # copy / subset edge attributes
        for key, value in store.items():
            if key == 'edge_index':
                continue

            if torch.is_tensor(value) and value.size(0) == store.edge_index.size(1):
                # edge-level attribute
                new_data[edge_type][key] = value[edge_mask]
            else:
                # global attribute for this relation
                new_data[edge_type][key] = value

    return new_data
