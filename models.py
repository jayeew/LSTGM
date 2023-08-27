#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Linear
from torch_geometric.utils import to_dense_adj
from utils import *
from layers import *

class LSTGM_layer(torch.nn.Module):
    def __init__(self, hidden, num_layers, N):
        super(LSTGM_layer, self).__init__()
        self.num_layers = num_layers
        self.N = N
        self.lstm = my_LSTM(input_size = hidden,
                        hidden_size = hidden,
                        num_layers = 1,
                        bias = True,
                        batch_first = False,
                        dropout = 0.0,
                        bidirectional = False)
        
    @classmethod
    def _norm(cls, edge_index):
        adj = to_dense_adj(edge_index).squeeze()
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        return adj_norm
    
    def forward(self, h, edge_index):
        adj_norm = self._norm(edge_index)
        h0 = h
        c0 = torch.zeros(h.size(), device=h.device)
        input = [adj_norm]
        for i in range(self.num_layers-1):
            temp = torch.matmul(input[-1], adj_norm)
            input.append(temp)
        input = torch.stack(input, dim=0)
        out, (h, c) = self.lstm(input, (h0, c0))
        h = torch.sum(out, dim=0)
        return h

class LSTGM(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(LSTGM, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.N = N
        self.lin_in = Linear(dataset.num_features, args.hidden)
        self.lin_out = Linear(args.hidden, dataset.num_classes)
        self.conv1 = LSTGM_layer(args.hidden, self.num_layers, N)
        self.conv2 = LSTGM_layer(args.hidden, self.num_layers, N)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_in.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.lin_out.weight, gain=math.sqrt(2))
        
        nn.init.constant_(self.lin_in.bias, 0)
        nn.init.constant_(self.lin_out.bias, 0)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        h = F.dropout(self.lin_in(x), p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        x = self.lin_out(h)

        return F.log_softmax(x, dim=1)
    