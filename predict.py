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
from word2tensor import load_data,numpy2tensors,labelconvert
import pickle
import pprint
import gensim
import multiprocessing
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
        model_file = open('80_trainingmodel256withrvec.bin', 'rb')
        param = pickle.load(model_file)
        model_file.close()
        encoderw, decoderw=param['encoder_w'],param['decoder_w']
        densew,denseb=param['dense_w'],param['dense_b']

        hiddensize=param['hidden_size']
        numstacks=param['num_stacks']
        drop_out=param['dropout']
        vocab_size=10000
        vocab_dim = 256
        cuda = device.create_cuda_gpu_on(1)
        encoder = layer.LSTM(name='lstm1', hidden_size=hiddensize, num_stacks=numstacks, dropout=drop_out, input_sample_shape=(vocab_dim,))
        decoder = layer.LSTM(name='lstm2', hidden_size=hiddensize, num_stacks=numstacks, dropout=drop_out, input_sample_shape=(vocab_dim,))
        encoder.to_device(cuda)
        decoder.to_device(cuda)
        print encoder.param_values()[0].shape,encoderw.shape
        encoder.param_values()[0].copy_from_numpy(encoderw, offset=0)
        decoder.param_values()[0].copy_from_numpy(decoderw, offset=0)

        dense = layer.Dense('dense', vocab_size, input_sample_shape=(hiddensize,))
        dense.to_device(cuda)
        dense.param_values()[0].copy_from_numpy(densew,offset=0)
        dense.param_values()[1].copy_from_numpy(denseb,offset=0)
        metadata,idx_q,idx_a=load_data()
        idx2w=metadata['idx2w']
        batcha=idx_a[300:301]
        ques = np.load('question_vec.npy')
        ans = np.load('answer_vec.npy')
        inputs=ques[300:301]
        inputs= numpy2tensors(inputs,dev=cuda)
        inputs.append(tensor.Tensor())
        inputs.append(tensor.Tensor())
        outputs = encoder.forward(model_pb2.kTrain, inputs)[-2:]



        model = gensim.models.Word2Vec.load("word2vec_gensim")
        start = model.wv['GO']
        start = start.reshape((1,1,256))
        start = numpy2tensors(start,cuda)
        start.extend(outputs)
        start=decoder.forward(model_pb2.kTrain,start)
        word = start[:-2][0]
        state = start[-2:]
        wlist=[]
        for i in range(20):
          nextword = dense.forward(model_pb2.kEval,word)
          nextw = tensor.to_numpy(nextword)
          wordvec = softmax(nextw[0])
          loca = np.argmax(wordvec)
          a = idx2w [loca]
          if (a != '_'):
             nword = model.wv[a]
             nword = nword.reshape((1,1,256))
          else:
             nword = np.zeros((1,1,256),dtype=np.float32)
          nword = numpy2tensors(nword,cuda)
          nword.extend(state)
          result = decoder.forward(model_pb2.kEval,nword)
          word = result[:-2][0]
          state = result[-2:]
          wlist.append(loca)
        print wlist
        print idx2w[700]
        print seq2word(wlist,idx2w)
