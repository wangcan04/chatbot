import cPickle as pickle
import numpy as np
import argparse
import pprint

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import loss
from singa import device
from singa import tensor
from singa import optimizer
from singa import initializer
from singa.proto import model_pb2
from tqdm import tnrange
from word2tensor import load_data,numpy2tensors,convert,labelconvert
import preprocess
import time

def softmax(x):
 """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def seq2word(x,dic):
    datalen=len(x)
    sen = []
    for i in range(datalen):
        a=x[i]
        sen.append(dic[a])
    return sen

if __name__ == "__main__":
        model_file = open('71.bin', 'rb')
        param = pickle.load(model_file)
        model_file.close()
        decoderw=param['decoder_w']
        densew,denseb=param['dense_w'],param['dense_b']
        hiddensize=param['hidden_size']
        numstacks=param['num_stacks']
        drop_out=param['dropout']
        vocab_size=7000
        cuda = device.create_cuda_gpu_on(1)
        encoder = layer.LSTM(name='lstm1', hidden_size=hiddensize, num_stacks=numstacks, dropout=drop_out, input_sample_shape=(vocab_size,))
        decoder = layer.LSTM(name='lstm2', hidden_size=hiddensize, num_stacks=numstacks, dropout=drop_out, input_sample_shape=(vocab_size,))
        encoder.to_device(cuda)
        decoder.to_device(cuda)
        encoder_w = encoder.param_values()[0]
        encoder_w.uniform(-0.08, 0.08)
        decoder.param_values()[0].copy_from_numpy(decoderw, offset=0)

        dense = layer.Dense('dense', vocab_size, input_sample_shape=(hiddensize,))
        dense.to_device(cuda)
        dense.param_values()[0].copy_from_numpy(densew,offset=0)
        dense.param_values()[1].copy_from_numpy(denseb,offset=0)
        
        
        metadata,idx_q,idx_a=load_data()
        idx2w=metadata['idx2w']
        batchq=idx_q[555:556]
        batcha=idx_a[555:556]

        inputs=convert(batchq,1,20,vocab_size,cuda)
        inputs.append(tensor.Tensor())
        inputs.append(tensor.Tensor())
        outputs = encoder.forward(model_pb2.kEval, inputs)[-2:]
        start = np.zeros((1,1,7000),dtype=np.float32)
        start[0,0,3] = 1
        start = numpy2tensors(start,cuda)
        start.extend(outputs)
        start=decoder.forward(model_pb2.kEval,start)
        word = start[:-2][0]
        state = start[-2:]
        wlist=[]
        for i in range(20):
          nextword = dense.forward(model_pb2.kEval,word)
          nextw = tensor.to_numpy(nextword)
          wordvec = softmax(nextw[0])
          loca = np.argmax(wordvec)
          nword = np.zeros((1,1,7000),dtype=np.float32)
          nword[0,0,loca]= 1
          nword = numpy2tensors(nword,cuda)
          nword.extend(state)
          result = decoder.forward(model_pb2.kEval,nword)
          word = result[:-2][0]
          state = result[-2:]
          wlist.append(loca)
        print wlist
        print seq2word(wlist,idx2w)
        print seq2word(batcha[0],idx2w)
