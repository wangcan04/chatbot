import numpy as np
from singa import tensor
import pickle
import gensim
import multiprocessing
from singa import device

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


def question2vec(i,max):
    model = gensim.models.Word2Vec.load("word2vec_gensim")
    lines = open("dataq.txt").read().split('\n')
    se = lines[i].split()
    vec = []
    if '' in se:
        se.remove('')
    for x in se:
        vec.append(model.wv[x])
    vec = np.array(vec)
    if len(vec) < max:
        pad = np.zeros((max, 100), dtype=np.float32)
        for i in range(len(vec)):
            pad[i] = vec[i]
        vec = pad
    return vec


def answer2vec(i,max):
    model = gensim.models.Word2Vec.load("word2vec_gensim")
    lines = open("dataa.txt").read().split('\n')
    se = lines[i].split()
    vec = []
    if '' in se:
        se.remove('')
    for x in se:
        vec.append(model.wv[x])
    vec = np.array(vec)
    if len(vec) < max:
        pad = np.zeros((max, 100), dtype=np.float32)
        for i in range(len(vec)):
            pad[i] = vec[i]
        vec = pad
    return vec


def numpy2tensors(num, dev):
    '''batch, seq, dim -- > seq, batch, dim'''
    tmpx = np.swapaxes(num, 0, 1)
    inputs = []
    for t in range(tmpx.shape[0]):
        x = tensor.from_numpy(tmpx[t])
        x.to_device(dev)
        inputs.append(x)
    return inputs



def labelconvert(batch,dev):
    return numpy2tensors(batch,dev)

'''
def convert(batch, batch_size, seq_length, vocab_size, dev):
    x = np.zeros((batch_size, seq_length, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        for t in range(seq_length):
            c = batch[b, t]
            x[b, t, c] = 1
    return numpy2tensors(x,dev)
'''
       
