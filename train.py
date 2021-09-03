#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:16:41 2020

@author: Xuye Liu
"""
import argparse
import os
import pickle
import random
import sys
import time
import traceback
import numpy as np
import logging
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K
from models.GCNLayer_pytorch import GraphConvolution
from timeit import default_timer as timer
from utils.myutils import batch_gen, init_tf, seq2sent
from models.HAConvGNN import HAConvGNN, TimeDistributed, Flatten
from utils.model import create_model
from utils.myutils import batch_gen, init_tf
from utils.timer import AverageMeter, AccuracyHelper
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()


def set_random_seed(seed = 10,deterministic=False,benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
def get_accuracy(scores, labels):
    correct = 0
    outputs = torch.argmax(scores, dim = 1)
    for predict, target in zip(outputs, labels):
        if predict == target:
            correct += 1
    return correct
    
if __name__ == '__main__':
#     wandb.init(project="notebook_test")
    set_random_seed(1337,deterministic=True)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('--epochs', dest='epochs', type=int, default=40)
    parser.add_argument('--data', dest='dataprep', type=str, default='../data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='./modelout')
    parser.add_argument('--asthops', dest='hops', type=int, default=2)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    asthops = args.hops

    # set gpu here
    init_tf(gpu)

    # Load tokenizers
    codetok = pickle.load(open('{}/code_notebook.tok'.format(dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('{}/coms_notebook.tok'.format(dataprep), 'rb'), encoding='UTF-8')
    asttok = pickle.load(open('{}/ast_notebook.tok'.format(dataprep), 'rb'), encoding='UTF-8')
    codevocabsize = codetok.vocab_size
    comvocabsize = comstok.vocab_size
    astvocabsize = asttok.vocab_size
    # TODO: setup config
    config = dict()
    config['asthops'] = asthops
    config['codevocabsize'] = codevocabsize
    config['comvocabsize'] = comvocabsize
    config['astvocabsize'] = astvocabsize
    # set sequence length for our input
    config['codelen'] = 200
    config['maxastnodes'] = 300
    config['comlen'] = 30
        
    config['batch_size'] = batch_size
    config['epochs'] = epochs

    # Load data
    seqdata = pickle.load(open('dataset_notebook.pkl', 'rb'))
    node_data = seqdata['strain_nodes']
    print("len",len(node_data))
    print("len",len(seqdata['sval_nodes']))
    edges = seqdata['strain_edges']
    config['edge_type'] = 'sml'
    test = seqdata['ctest']
    dttest = seqdata['dttest']
    # model parameters
    steps = int(len(seqdata['ctrain'])/batch_size)
#     steps = 50
#     valsteps = 50
    valsteps = int(len(seqdata['cval'])/batch_size)


    # Print information
    print('codevocabsize {}'.format(codevocabsize))
    print('comvocabsize {}'.format(comvocabsize))
    print('astvocabsize {}'.format(astvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*batch_size))
    print('------------------------------------------')
    # create model
    net, device = create_model(config)
    optimizer = torch.optim.Adamax(net.parameters(), lr = 2e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    # set up data generators

    train_data = Data.DataLoader(dataset = seqdata['ctrain'], batch_size=200, shuffle = True)
    valid_data = Data.DataLoader(dataset = seqdata['cval'], batch_size= 200, shuffle = True)
    train_gen = batch_gen(seqdata, 'train', config, nodedata=node_data, edgedata=edges)
    valgen = batch_gen(seqdata, 'val', config, nodedata=seqdata['sval_nodes'], edgedata=seqdata['sval_edges'])
    testgen = batch_gen(seqdata, 'test', config, nodedata=seqdata['stest_nodes'], edgedata=seqdata['stest_edges'])
    outfn = outdir+"/predictions/loss_notebook.txt"
    outf = open(outfn, 'w')
    start_epoch = 0
    stats = {'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}
    ## Train
    history_acc = []
    history_loss = []
    history_valacc = []
    hitory_valloss = []
    for epoch in range(70):
        train_loss, valid_loss = [], []
        total_correct = 0.0
        ## training part
        bar = tqdm(range(steps))
        val_bar = tqdm(range(valsteps))
        ml_loss = AverageMeter()
        acc = AccuracyHelper()
        val_loss = AverageMeter()
        val_acc = AccuracyHelper()
        bar.set_description("%s" % 'Epoch = %d [accuracy = x.xx, loss = x.xx]' %
                        epoch)
        val_bar.set_description("%s" % 'Epoch = %d [accuracy = x.xx, loss = x.xx]' %
                        epoch)
        for step in bar:
            train_batch = train_gen.getitem(step)
            train1 = train_batch[0]  ##tdatseqs, comseqs, smlnodes, wedge_1
            train2 = train_batch[1]  ##comouts
            train2 = np.array(train2)
            train2 = torch.from_numpy(train2)
            train2 = train2.type(torch.LongTensor)
            train2 = train2.to(device)
            for i in range(4):
                train1[i] = np.array(train1[i])
                train1[i] = torch.from_numpy(train1[i])
                train1[i] = train1[i].type(torch.LongTensor)
                train1[i] = train1[i].to(device)
            output = net([train1[0], train1[1], train1[2], train1[3]])
            loss = loss_func(output, train2)
            total_correct += get_accuracy(output, train2)
            optimizer.zero_grad()
            # backward propogation
            loss.backward()
            # weight optimizer
            optimizer.step()
            ml_loss.update(loss, batch_size*30)
            acc.update(total_correct, batch_size * 30)
            log_info = 'Epoch = %d [accuracy = %.4f, loss = %.4f]' % \
                (epoch, acc.avg, ml_loss.avg)
            bar.set_description("%s" % log_info)
            train_loss.append(loss.item())
        outf.write("loss" + str(np.mean(train_loss)) + "accuracy" + str(acc.avg) + "\n" )
        outf.flush()
        total_correct = 0.0
        with torch.no_grad():
            for step in val_bar:
                val_batch = valgen.getitem(step)
                val1 = val_batch[0]  ##tdatseqs, comseqs, smlnodes, wedge_1
                val2 = val_batch[1]  ##comouts
                val2 = np.array(val2)
                val2 = torch.from_numpy(val2)
                val2 = val2.type(torch.LongTensor)
                val2 = val2.to(device)
                for i in range(4):
                    val1[i] = np.array(val1[i])
                    val1[i] = torch.from_numpy(val1[i])
                    val1[i] = val1[i].type(torch.LongTensor)
                    val1[i] = val1[i].to(device)
                output = net([val1[0], val1[1], val1[2], val1[3]])
                loss = loss_func(output, val2)
                total_correct += get_accuracy(output, val2)
                val_loss.update(loss, batch_size*30)
                val_acc.update(total_correct, batch_size*30)
                log_info = 'Epoch = %d [val_acc = %.4f, val_loss = %.4f]' % \
                    (epoch, val_acc.avg, val_loss.avg)
                val_bar.set_description("%s" % log_info)
                valid_loss.append(loss.item())
        history_acc.append(acc.avg)
        history_loss.append(ml_loss.avg)
        history_valacc.append(val_acc.avg)
        hitory_valloss.append(val_loss.avg)
        outf.write("valid loss" + str(np.mean(valid_loss)) + "val_acc" + str(val_acc.avg) + "\n" )
        outf.flush()
        train_gen.on_epoch_end()
        valgen.on_epoch_end()
        if val_acc.avg > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d)' %
                        ("val_acc", val_acc.avg, epoch))
            torch.save({'epoch': epoch,'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),'loss': loss}, 
                        "./modelout/" +"HAConvGNN_saved_model.h5")
            print("Improved from %.4f to %.4f" % (stats['best_valid'], val_acc.avg))
            stats['best_valid'] = val_acc.avg
            stats['no_improvement'] = 0
        else:
            print("No improvement, best is %.4f" % (stats['best_valid']))
            stats['no_improvement'] += 1
            if stats['no_improvement'] >= 2:
                break
        print("Epoch: ", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss", np.mean(valid_loss))
    outf.close()
    ## Test
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    plt.plot(history_acc)
    plt.plot(history_valacc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    
    plt.subplot(212)
    plt.plot(history_loss)
    plt.plot(hitory_valloss)
    plt.yscale('log')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.show()
    plt.savefig('./plot.png')
