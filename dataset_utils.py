#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch_geometric.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB

def DataLoader(name):

    name = name.lower()
    root_path = 'D:/tmp/'
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root_path, name, split='random', num_train_per_class=20, num_val=500, num_test=1000, transform=T.NormalizeFeatures())
    
    elif name in ['computers', 'photo']:
        dataset = Amazon(root_path, name, T.NormalizeFeatures())

    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return dataset, data

    elif name in ['film']:
        dataset = Actor(root=root_path+'Actor', transform=T.NormalizeFeatures())
        
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=root_path, name=name, transform=T.NormalizeFeatures())
     
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset, dataset[0]
