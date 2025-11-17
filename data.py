import torch
import torch.nn.functional as F
import torch_geometric as pyg
from slop import subset_ogbn_mag_to_train_papers

metadata = (['paper', 'author', 'institution', 'field_of_study'], [('author', 'affiliated_with', 'institution'), ('author', 'writes', 'paper'), ('paper', 'cites', 'paper'), ('paper', 'has_topic', 'field_of_study'), ('institution', 'rev_affiliated_with', 'author'), ('paper', 'rev_writes', 'author'), ('paper', 'rev_cites', 'paper'), ('field_of_study', 'rev_has_topic', 'paper')])

def load_dataset(remove_test=False):
    dataset = pyg.datasets.OGB_MAG("./data")[0]
    if remove_test:
        dataset = subset_ogbn_mag_to_train_papers(dataset)
    dataset = pyg.transforms.ToUndirected(merge=False)(dataset)
    return dataset
