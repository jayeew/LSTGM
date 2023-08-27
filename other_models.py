#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn.functional as F
import torch_sparse
import numpy as np
import math
from torch import FloatTensor
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear, GRU, ModuleList, Conv1d
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, FAConv, APPNP, SAGEConv, SGConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse, degree, add_remaining_self_loops
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric.transforms as T
from torch_scatter import scatter_add
from torch_cluster import random_walk

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self,  args, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = args.num_layers
        self.alpha = 0.9

        # PPR-like
        TEMP = self.alpha * (1 - self.alpha) ** np.arange(self.K + 1)
        TEMP[-1] = (1 - self.alpha) ** self.K
        
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args)

        self.dprate = 0.5
        self.dropout = 0.5

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            h = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            h = self.prop1(x, edge_index)

            return F.log_softmax(x, dim=1), h

class MLP(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)

        self.dropout = 0.5

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1), None

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(GCN_Net, self).__init__()
        self.conv1 = SGConv(dataset.num_features, args.hidden)
        # self.conv2 = SGConv(args.hidden, args.hidden)
        self.num_layers = args.num_layers
        self.convs = ModuleList()
        for i in range(args.num_layers-1):
            self.convs.append(GCNConv(args.hidden, args.hidden))
        self.dropout = args.dropout
        self.lin1 = nn.Linear(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        # self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers-1):
            h = self.convs[i](x, edge_index)
            x = F.dropout(F.relu(x+h))
        # h = self.conv2(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, args.hidden)
        self.conv2 = SAGEConv(args.hidden, dataset.num_classes)
        self.num_layers = args.num_layers
        self.convs = ModuleList()
        # for i in range(args.num_layers):
        #     self.convs.append(GCNConv(args.hidden, args.hidden))
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # for i in range(self.num_layers):
        #     x = self.convs[i](x, edge_index)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), None

class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=8,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * 8,
            dataset.num_classes,
            heads=1,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), None

class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(10, 1)
        self.dropout = 0.5

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1), None

class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args, N):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=64,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1), None       

class H2GCN_Net(torch.nn.Module):

    def __init__(self, dataset, args, N):
        super(H2GCN_Net, self).__init__()
        dropout = args.dropout
        hidden_dim = args.hidden
        feat_dim = dataset.num_features
        class_dim = dataset.num_classes
        k = 2

        self.dropout = dropout
        self.k = k
        self.w_embed = torch.nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = torch.nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.w_embed)
        torch.nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, data):
        adj, x = data.edge_index, data.x  # (adj: torch.sparse.Tensor, x: FloatTensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [F.relu(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(torch.cat([r1, r2], dim=1))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return F.log_softmax(torch.mm(r_final, self.w_classify), dim=1), None

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, dataset, args, N):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(GraphConvolution(args.hidden, args.hidden,variant=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(dataset.num_features, args.hidden))
        self.fcs.append(nn.Linear(args.hidden, dataset.num_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.alpha = 0.1
        self.lamda = 0.5

    def forward(self, data):
        _layers = []
        x, edge_index = data.x, data.edge_index
        adj = to_dense_adj(edge_index).squeeze()
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1), None

class FAGCN_Net(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(FAGCN_Net, self).__init__()
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        self.eps = 0.1
        self.layer_num = args.num_layers
        self.dropout = args.dropout
        self.hidden = args.hidden

        self.layers = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(self.hidden, self.eps, self.dropout))

        self.lin1 = torch.nn.Linear(in_channels, self.hidden)
        self.lin2 = torch.nn.Linear(self.hidden, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, raw, edge_index)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1), None

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)#(n, k+1, c)
        retain_score = self.proj(pps)#(n, k+1, 1)
        retain_score = retain_score.squeeze()#(n, k+1)
        retain_score = torch.sigmoid(retain_score)#(n, k+1)
        retain_score = retain_score.unsqueeze(1)#(n, 1, k+1)
        out = torch.matmul(retain_score, pps).squeeze()#(n, c)
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()
        
class DAGNN(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(DAGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = Prop(dataset.num_classes, args.num_layers)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1), None