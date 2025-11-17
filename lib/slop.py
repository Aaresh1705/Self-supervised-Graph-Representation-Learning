#!/usr/bin/env python3

import torch
from torch_geometric.data import HeteroData

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
