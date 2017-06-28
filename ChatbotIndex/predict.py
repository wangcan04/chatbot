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
        metadata,idx_q,idx_a=load_data()
        batchq=idx_q[300:301]
        inputs=convert(batchq,1,20,vocab_size,cuda)
        inputs.append(tensor.Tensor())
        inputs.append(tensor.Tensor())
        outputs = encoder.forward(model_pb2.kTrain, inputs)[-2:]

        start = np.zeros((1,1,7000),dtype=np.float32)
        start[0,0,3] = 1
        print start
        result = numpy2tensors(start,cuda)
        result.extend(outputs)
        result=decoder.forward(model_pb2.kTrain,result)
        words = result[:-2]
        state = result[-2:]
        for word in words:
            nextword = dense.forward(model_pb2.kTrain,word)
        nextw = tensor.to_numpy(nextword)
        print max(nextw[0]),min(nextw[0])
        a = softmax(nextw[0])
        print max(a)

