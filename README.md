# Self-Supervised Graph Representation Learning

This repository contains the implementation and experiments for the project **Self-Supervised Graph Representation Learning**, comparing supervised, self-supervised, and multi-task learning approaches on heterogeneous graph data.

##  Project Overview
We evaluate 3 architectures:
- **Supervised GraphSAGE**
- **Self-supervised GraphMAE**
- **Multi-Task Learning (MTL)** combining node and edge prediction from supervised

The models are trained and tested on the **OGB-MAG** dataset and evaluated on:
- **Node classification** (venue prediction)
- **Link prediction** (paper-field edges)

## Key Findings
- Self-supervised GraphMAE produces embeddings competitive with supervised GraphSAGE.
- Multi-task learning improves node classification but slightly reduces link prediction performance.
- High AUC scores are influenced by strong class imbalance (large number of negative edges).

## Repository Structure
```
/lib/                       # Our own python package where we store essentail functions and classes
/mag/                       # Processed OGB-MAG data and metapath2vec embeddings
/supervised/                # All train, test, models, and figures for supervised models
/supervised/models/         # Best model for supervised and MTL models
/batch/                     # Outputs and metrics from HPC
```

## Report
The full project report is included as `Self_supervised_Graph_Representation_Learning.pdf`.
