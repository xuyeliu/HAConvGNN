import random
import sys
from timeit import default_timer as timer

import keras
import numpy as np
import copy

start = 0
end = 0

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()
def init_tf(gpu):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))

class batch_gen():
    def __init__(self, seqdata, tt, config, nodedata=None, edgedata=None):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']
        self.seqdata = seqdata
        self.allfids = list(seqdata['dt%s' % (tt)].keys())
        self.config = config
        self.edgedata = edgedata
        self.nodedata = nodedata
        random.shuffle(self.allfids)

    def getitem(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfids = self.allfids[start:end]
        return self.make_batch(batchfids)

    def make_batch(self, batchfids):
        return self.divideseqs(batchfids, self.seqdata, self.nodedata, self.edgedata, self.comvocabsize, self.tt)
        

    def __len__(self):
        return int(np.ceil(len(list(self.seqdata['dt%s' % (self.tt)]))/self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allfids)

    def divideseqs(self, batchfids, seqdata, nodedata, edge1, comvocabsize, tt):
        import keras.utils

        codeseqs = list()
        comseqs = list()
        astnodes = list()

        wedge_1 = list()

        comouts = list()

        fiddat = dict()

        for fid in batchfids:
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            try:
                wastnodes = nodedata[fid]
            except:
                continue

            try:
                edge_1 = copy.copy(edge1[fid])

            except:
                continue
            newlen = 4 - len(wastnodes)
            if newlen < 0:
                newlen = 0
            wastnodes = wastnodes.tolist()
            for k in range(newlen):
                wastnodes.append(np.zeros(300, dtype='int32'))
            for i in range(0, len(wastnodes)):
                wastnodes[i] = np.array(wastnodes[i])[:300]
                tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
                tmp[:wastnodes[i].shape[0]] = wastnodes[i]
                wastnodes[i] = np.int32(tmp)
            wastnodes = np.int32(wastnodes)
            wastnodes = np.asarray(wastnodes)
            wastnodes = wastnodes[:4,:]
            newlen = 4 - len(edge_1)
            if newlen < 0:
                newlen = 0
            for k in range(newlen):
                edge_1.append(np.zeros((300, 300), dtype='int32'))
            if newlen > 0:
                for i in range(0, 4 - newlen):
                    edge_1[i] = np.asarray(edge_1[i].todense())
                    edge_1[i] = edge_1[i][:self.config['maxastnodes'], :self.config['maxastnodes']]
                    tmp_1 = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
                    tmp_1[:edge_1[i].shape[0], :edge_1[i].shape[1]] = edge_1[i]
                    edge_1[i] = np.int32(tmp_1)
            else:
                for i in range(0, len(edge_1)):
                    edge_1[i] = np.asarray(edge_1[i].todense())
                    edge_1[i] = edge_1[i][:self.config['maxastnodes'], :self.config['maxastnodes']]
                    tmp_1 = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
                    tmp_1[:edge_1[i].shape[0], :edge_1[i].shape[1]] = edge_1[i]
                    edge_1[i] = np.int32(tmp_1)
            edge_1 = np.array(edge_1)
            edge_1 = edge_1[:4,:, :]

            wtdatseq = wtdatseq[:self.config['codelen']]
            if tt == 'test':
                fiddat[fid] = [wtdatseq, wcomseq, wastnodes, edge_1]
            else:
                for i in range(0, len(wcomseq)):
                    codeseqs.append(wtdatseq)
                    astnodes.append(wastnodes)
                    wedge_1.append(edge_1)
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]

                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(comout)

 
        codeseqs = np.asarray(codeseqs)

        astnodes = np.asarray(astnodes)

        wedge_1 = np.asarray(wedge_1)

        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)
        if tt == 'test':
            return fiddat
        else:
            # if self.config['num_output'] == 2:
            # return [[codeseqs, comseqs, astnodes, wedge_1],
            #         [comouts, comouts]]
            # else:
            #     if (self.config['use_tdats']):
            return [[codeseqs, comseqs, astnodes, wedge_1],
                    comouts]
            #     else:
            # return [[comseqs, astnodes, wedge_1], comouts]

