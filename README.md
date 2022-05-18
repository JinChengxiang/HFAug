# README

# HFAug

This repository provides a reference implementation of MAGNN as described in the paper:

> HFAug: Heterogeneous Feature Augmentation for Ponzi Detection in Ethereum
> 

Available at [link](https://arxiv.org/abs/2204.08916).

# Dependencies

Recent versions of the following packages for Python are required:

- dgl-cuda10.2  0.8.0
- pytoch  1.10.0
- scikit-learn  1.0.1
- toch-geometric-temporal  0.50.0
- numpy  1.20.3

# Data

In `/data/data_het`, there are data about building heterogeneous graph, and others in `/data` are used in homogeneous graph. 

# Usage

1. Create metapaths with `/data/data_het/dgl_metapath.py` and embedding with `/random_walk/embedding.py`
2. Use *HFAug* in different methods. Use `HFAug_ML.py` for manual feature, `/random_walk/HFAug_RW.py` for embedding and GNNs in `/GNN/{methods}`
3. Repeat and report the best one