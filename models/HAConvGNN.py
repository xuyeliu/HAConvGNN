
import random
import string
from typing import Union, List
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.GCNLayer_pytorch import GraphConvolution
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        if len(x.size()) == 4:
            t, n = x.size(0), x.size(1)
            x_reshape = x.contiguous().view(t, n, -1)
            y, y_state = self.module(x_reshape)
            y = y.contiguous().view(t, n, y.size()[-1])
            return y
        t, n = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, y.size()[1])
        return y

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `query`: (`batch_size`, #queries, `d`)
    # `key`: (`batch_size`, #kv_pairs, `d`)
    # `value`: (`batch_size`, #kv_pairs, `dim_v`)
    # `valid_len`: either (`batch_size`, ) or (`batch_size`, xx)
    def forward(self, query, key, valid_len=None):
        d = query.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of key
        scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(d)
        attention_weights = self.dropout(nn.functional.softmax(scores, dim=-1))
        return torch.bmm(attention_weights, key)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return output

class HAConvGNN(nn.Module):

    def __init__(self, config, device):
        super(HAConvGNN, self).__init__()
        self.name = 'HAConvGNN'
        self.embdims = 300
        self.smldims = 256
        self.recdims = 256
        self.tdddims = 256
        self.contextdims = 256
        self.config = config
        self.codevocabsize = int(config['codevocabsize'])
        self.comvocabsize = int(config['comvocabsize'])
        self.astvocabsize = int(config['astvocabsize'])
        self.codelen = int(config['codelen'])
        self.comlen = int(config['comlen'])
        self.astlen = int(config['maxastnodes'])
        self.activation1 = torch.nn.DataParallel(nn.Softmax(dim = 2))
        self.activation2 = torch.nn.DataParallel(nn.Softmax(dim = 2))
        self.activation3 = torch.nn.DataParallel(nn.ReLU())
        self.activation4 = torch.nn.DataParallel(nn.ReLU())
        self.config['batch_maker'] = 'graph_multi_1'
        self.embed_tda = torch.nn.DataParallel(nn.Embedding(self.astvocabsize, self.embdims))
        self.embed_com = torch.nn.DataParallel(nn.Embedding(self.comvocabsize, self.embdims))
        self.gru1 = torch.nn.DataParallel(nn.GRU(input_size=self.embdims, 
                          hidden_size=self.recdims, 
                          num_layers=1,
                          batch_first=True))
        self.gru2 = torch.nn.DataParallel(nn.GRU(input_size=self.embdims, 
                          hidden_size=self.recdims, 
                          num_layers=1,
                          batch_first=True))
        self.gru3 = torch.nn.DataParallel(nn.GRU(input_size=self.embdims, 
                          hidden_size=self.recdims, 
                          num_layers=1,
                          batch_first=True))
        self.gru4 = torch.nn.DataParallel(nn.GRU(input_size=90000, 
                          hidden_size=self.recdims, 
                          num_layers=1,
                          batch_first=True))
        self.gru5 = torch.nn.DataParallel(nn.GRU(input_size=300, 
                              hidden_size=self.recdims, 
                              num_layers=1,
                              batch_first=True))
        self.gru6 = torch.nn.DataParallel(nn.GRU(input_size=300, 
                              hidden_size=self.recdims, 
                              num_layers=1,
                              batch_first=True))
        self.gru7 = torch.nn.DataParallel(nn.GRU(input_size=300, 
                              hidden_size=self.recdims, 
                              num_layers=1,
                              batch_first=True))
        self.gru8 = torch.nn.DataParallel(nn.GRU(input_size=300, 
                              hidden_size=self.recdims, 
                              num_layers=1,
                              batch_first=True))
        self.gcnlayer = torch.nn.DataParallel(GraphConvolution(300, 300, device))
        self.dense = torch.nn.DataParallel(nn.Linear(768, 256))
        self.Flatten = torch.nn.DataParallel(Flatten())
        self.linear = torch.nn.DataParallel(nn.Linear(7680, self.comvocabsize))
        self.attention1 = torch.nn.DataParallel(DotProductAttention(dropout=0.5))
        self.attention2 = torch.nn.DataParallel(DotProductAttention(dropout=0.5))
        self.attention3 = torch.nn.DataParallel(DotProductAttention(dropout=0.5))
        self.attention4 = torch.nn.DataParallel(DotProductAttention(dropout=0.5))
        
    def forward(self, x):
        tde = self.embed_tda(x[0])
        
        se = self.embed_tda(x[2])
        tde = tde.type(torch.FloatTensor)
        se = se.type(torch.FloatTensor)
        tencout, tstate_h = self.gru1(tde)
        de = self.embed_com(x[1])
        de = de.type(torch.FloatTensor)
        decout, destate = self.gru2(de, tstate_h)
        tcontext = self.attention1(decout, tencout)
        astwork = se
        astwork = astwork.permute(1, 0, 2, 3)
        tmp = astwork
        x[3] = x[3].permute(1, 0, 2, 3)
        for i in range(self.config['asthops']):
            for j in range(4):
                astwork_tmp = self.gcnlayer(astwork[j], x[3][j])
                astwork[j] = self.activation4(astwork_tmp)
        astwork1 = astwork
        astwork = TimeDistributed(self.gru4)(tmp)
        d = decout.shape[-1]
        astwork = astwork.permute(1, 0, 2)
        scores = torch.bmm(decout, astwork.transpose(1,2)) / math.sqrt(d)
        ahierarchy = nn.functional.softmax(scores, dim=-1)
        ast_list = []
        m = 0
        ahierarchy = ahierarchy.permute(2, 0, 1)
        for k in astwork1:
            if m == 0:
                output, state = self.gru5(k, tstate_h)
            elif m == 1:
                output, state = self.gru6(k, tstate_h)
            elif m == 2:
                output, state = self.gru7(k, tstate_h)
            elif m == 3:
                output, state = self.gru8(k, tstate_h)
            scores = torch.bmm(decout, output.transpose(1,2)) / math.sqrt(d)
            asingle_hierarchy = nn.functional.softmax(scores, dim=-1)
            new_atten = element_wise_mul(asingle_hierarchy, ahierarchy[m])
            output = torch.bmm(new_atten, output)
            output = output.unsqueeze(0)
            ast_list.append(output)
            m += 1
        m = 0
        output = torch.cat(ast_list, 0)
        acontext = torch.sum(output, 0).squeeze(0)
        acontext = self.attention4(decout, acontext)
        context = torch.cat([tcontext, decout, acontext], 2)
        out = self.dense(context)
        out = self.activation3(out)
        out = self.Flatten(out)
        out1 = self.linear(out)
        return out1
    