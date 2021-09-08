#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:16:41 2020

@author: Xuye Liu
"""
# from .models.codegnngru import CodeGNNGRU
import argparse
import os
import pickle
import random
import sys
import time
import traceback
import numpy as np
# import tensorflow as tf
import torch
# from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
# from torch_scatter import scatter_add
# from torch_geometric.nn.conv import MessagePassing
# from keras.callbacks import ModelCheckpoint, Callback
# import keras.backend as K
from models.GCNLayer_pytorch import GraphConvolution
from timeit import default_timer as timer
from utils.myutils import batch_gen, init_tf, seq2sent
from models.HAConvGNN import HAConvGNN, TimeDistributed, Flatten
from utils.model import create_model
from utils.myutils import batch_gen, init_tf

def set_random_seed(seed = 10,deterministic=False,benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
def gen_pred(model, data, device, comstok, comlen, batchsize, config, fid_set, strat='greedy'):
    tdats, coms, wsmlnodes, wedge_1 = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)
    tdats = torch.from_numpy(tdats)
    coms = torch.from_numpy(coms)
    wsmlnodes = torch.from_numpy(wsmlnodes)
    wedge_1 = torch.from_numpy(wedge_1)
    tdats = tdats.type(torch.LongTensor)
    coms = coms.type(torch.LongTensor)
    wsmlnodes = wsmlnodes.type(torch.LongTensor)
    wedge_1 = wedge_1.type(torch.LongTensor)
    tdats = tdats.to(device)
    coms = coms.to(device)
    wsmlnodes = wsmlnodes.to(device)
    wedge_1 = wedge_1.to(device)
    for i in range(1, comlen):
        if i == 1:
            pass
        else:
            coms = torch.from_numpy(coms)
            coms = coms.type(torch.LongTensor)
        output = model([tdats, coms, wsmlnodes, wedge_1])
        output = output.cpu().detach().numpy()
        coms = coms.cpu().numpy()
        for c, s in enumerate(output):
            coms[c][i] = np.argmax(s)
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', type=str, default=None)
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='../data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='modelout/')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=2) 
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)

    args = parser.parse_args()

    modelfile = args.model
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batchsize = args.batchsize
    outfile = args.outfile

    config = dict()

    # User set parameters#
    config['maxastnodes'] = 300
    config['asthops'] = 2

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    codetok = pickle.load(open('{}/code_notebook.tok'.format(dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('{}/coms_notebook.tok'.format(dataprep), 'rb'), encoding='UTF-8')
    asttok = pickle.load(open('{}/ast_notebook.tok'.format(dataprep), 'rb'), encoding='UTF-8')
   
    seqdata = pickle.load(open('dataset_notebook.pkl', 'rb'))

    allfids = list(seqdata['ctest'].keys())
    codevocabsize = codetok.vocab_size
    comvocabsize = comstok.vocab_size
    astvocabsize = asttok.vocab_size

    config['codevocabsize'] = codevocabsize
    config['comvocabsize'] = comvocabsize
    config['astvocabsize'] = astvocabsize
    print('codevocabsize {}'.format(codevocabsize))
    print('comvocabsize {}'.format(comvocabsize))
    print('astvocabsize {}'.format(astvocabsize))

    # set sequence lengths
    config['codelen'] = 200
    config['comlen'] = 30
    config['batch_size'] = batchsize

    comlen = 30
    print('len', len(seqdata['ctest']))
    print('allfids', len(allfids))
    model, device = create_model(config)
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adamax(model.parameters(), lr = 1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_func = torch.nn.CrossEntropyLoss()
    print("MODEL LOADED")

    node_data = seqdata['stest_nodes']
    edgedata = seqdata['stest_edges']
    config['batch_maker'] = 'graph_multi_1'
    testgen = batch_gen(seqdata, 'test', config, nodedata=seqdata['stest_nodes'], edgedata=seqdata['stest_edges'])

    print(model)

    # set up prediction string and output file
    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict_notebook.txt"
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
 
    #predict
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        for fid in fid_set:
            seqdata['ctest'][fid] = comstart #np.asarray([stk])
        batch = testgen.make_batch(fid_set)
        batch_results = gen_pred(model, batch, device, comstok, comlen, batchsize, config, fid_set, strat='greedy')
        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))
            outf.flush()
        
        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, int(batchsize/(end-st))), end='\r')
    outf.close()        

