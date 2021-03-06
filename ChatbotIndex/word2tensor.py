import numpy as np
from singa import tensor
import pickle

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


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
def convert(batch, batch_size, seq_length, vocab_size, dev):
    '''convert a batch of data into a sequence of input tensors'''
    x = np.zeros((batch_size, seq_length, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        for t in range(seq_length):
            c = batch[b, t]
            x[b, t, c] = 1
    return numpy2tensors(x,dev)
