#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:16:41 2020

@author: Xuye Liu
"""
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, device, activation='relu', initializer='glorot_uniform', sparse=False, use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer
        self.sparse = sparse
        self.use_bias = use_bias
        self.device = device
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))           # in_features × out_features
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input, adj):
        input = input.to(self.device)
        adj = adj.to(self.device)
        adj += torch.eye(int(adj.shape[1])).type(torch.LongTensor).to(self.device)
        output = torch.matmul(adj.type(torch.FloatTensor).to(self.device), input).to(self.device)             # torch.spmm：稀疏矩阵乘法，sp即sparse。
        output = torch.matmul(output, self.weight.to(self.device))
        if self.bias is not None:
            return output + self.bias.to(self.device)
        else:
            return output


    '''showing objkect in string type, showing in the console'''
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
