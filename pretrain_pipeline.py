#!/usr/bin/env python3
import lib
from lib.model import pretrain_gmae
from lib.dataset import load_data

from torch_geometric.transforms import Compose, ToUndirected

transform = Compose([ToUndirected(merge=False)])
preprocess = 'metapath2vec'
data = lib.dataset.load_data("", transform=transform, preprocess=preprocess)

pretrain_gmae(data)
