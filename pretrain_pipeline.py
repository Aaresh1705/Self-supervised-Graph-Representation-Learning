#!/usr/bin/env python3
import lib
from lib.model import pretrain_gmae
from lib.dataset import load_data
import torch

from torch_geometric.transforms import Compose, ToUndirected

model_type = "gmae" # or gae

pretrain_function = pretrain_gmae if model_type  == "gmae" else pretrain_gae

transform = Compose([ToUndirected(merge=False)])
preprocess = 'metapath2vec'
data = lib.dataset.load_data("OGBN-MAG/", transform=transform, preprocess=preprocess)

train_data = data.subgraph({
    "paper": data["paper"].train_mask.nonzero(as_tuple=False).view(-1)
})

encoder = pretrain_function(train_data)

torch.save(encoder.state_dict(), model_type + "_encoder")
