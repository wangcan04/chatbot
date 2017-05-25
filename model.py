import cPickle as pickle
import numpy as np
import argparse

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import loss
from singa import device
from singa import tensor
from singa import optimizer
from singa import initializer
from singa.proto import model_pb2
from tqdm import tnrange
from word2vector import numpy2tensors,answer2vec,question2vec



def get_lr(epoch):
    return 0.001 / float(1 << (epoch / 50))

# SGD with L2 gradient normalization
opt = optimizer.RMSProp(constraint=optimizer.L2Constraint(5))
cuda = device.get_default_device()
encoder = layer.LSTM(name='lstm', hidden_size=32, num_stacks=1, dropout=0.5, input_sample_shape=(100,))
decoder = layer.LSTM(name='lstm', hidden_size=32, num_stacks=1, dropout=0.5, input_sample_shape=(100,))
encoder.to_device(cuda)
decoder.to_device(cuda)
encoder_w = encoder.param_values()[0]
encoder_w.uniform(-0.08, 0.08)
decoder_w = decoder.param_values()[0]
decoder_w.uniform(-0.08, 0.08)

lossfun = loss.SoftmaxCrossEntropy()
batch_size=40
train_loss = 0
maxlength=20
for epoch in range(20):
    bar = range(50)
    for b in bar:
        inputbatch = []
        labelbatch=[]
        for i in range(b*batch_size,(b+1)*batch_size):
             inputbatch.append(question2vec(i,maxlength))
        inputbatch=np.array(inputbatch)
        inputs= numpy2tensors(inputbatch,dev=device.get_default_device())
        inputs.append(tensor.Tensor())
        inputs.append(tensor.Tensor())
        for i in range(b*batch_size,(b+1)*batch_size):
             labelbatch.append(question2vec(i,maxlength))
        labelbatch=np.array(labelbatch)
        labels= numpy2tensors(labelbatch,dev=device.get_default_device())
        labels.append(tensor.Tensor())
        labels.append(tensor.Tensor())
