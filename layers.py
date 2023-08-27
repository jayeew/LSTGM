#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
from torch.nn import Parameter
import math

class my_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, 
                 batch_first = False, dropout = 0.0, bidirectional = False):
        super(my_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        #i_t
        self.Whi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = Parameter(torch.Tensor(hidden_size))
        
        #f_t
        self.Whf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = Parameter(torch.Tensor(hidden_size))
        
        #g_t
        self.Whg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = Parameter(torch.Tensor(hidden_size))
        
        #o_t
        self.Who = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = Parameter(torch.Tensor(hidden_size))

        self.init_weight()
        
    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def conv(self, adj, h):
        h = adj @ h
        return h
    
    def forward(self, x, hx=None):
        L, N, hidden = x.size()
        if hx is None:
            h_t = torch.zeros(N, hidden, dtype=x.dtype, device=x.device)
            c_t = torch.zeros(N, hidden, dtype=x.dtype, device=x.device)
        else:
            h_t, c_t = hx
        h_0 = h_t
        out = [h_0]
        for t in range(L):
            x_t = x[t, :, :]

            f_t = self.conv(x_t, h_t)
            f_t = torch.sigmoid(torch.matmul(f_t, self.Whf) + self.bf)

            i_t = h_t
            i_t = torch.sigmoid(torch.matmul(i_t, self.Whi) + self.bi)

            g_t = h_t
            g_t = torch.tanh(torch.matmul(g_t, self.Whg) + self.bg)

            o_t = self.conv(x_t, h_t)
            o_t = torch.sigmoid(torch.matmul(o_t, self.Who) +self.bo)
            
            c_t = torch.mul(f_t, c_t) + self.conv(x_t, torch.mul(i_t, g_t))
            h_t = torch.mul(o_t, torch.tanh(c_t))

            out.append(h_t)

        out = torch.stack(out, dim=0)
        return out, (h_t, c_t)
