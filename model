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
encoder = layer.LSTM(name='lstm', hidden_size=32, num_stacks=1, dropout=0.5, input_sample_shape=100)
decoder = layer.LSTM(name='lstm', hidden_size=32, num_stacks=1, dropout=0.5, input_sample_shape=100)
encoder.to_device(cuda)
decoder.to_device(cuda)
encoder_w = encoder.param_values()[0]
encoder_w.uniform(-0.08, 0.08)
decoder_w = decoder.param_values()[0]
decoder_w.uniform(-0.08, 0.08)

lossfun = loss.SoftmaxCrossEntropy()
batch_size=20
seq_length=20
train_loss = 0
for epoch in range(50):
    inputbatch = []
    labelbatch=[]
    bar = tnrange(50000, desc='Epoch %d' % 0)
    for b in bar:
        for i in range(b*batch_size,(b+1)*batch_size):
             inputbatch.append(question2vec(i,max))
        inputbatch=np.array(inputbatch)
        inputs= numpy2tensors(inputbatch,dev=device.create_cuda_gpu())
        for i in range(b*batch_size,(b+1)*batch_size):
             labelbatch.append(question2vec(i,max))
        labelbatch=np.array(labelbatch)
        labelsinput =  numpy2tensors(labelbatch, dev=device.create_cuda_gpu())
        inputs.append(tensor.Tensor())
        labelsinput.append(tensor.Tensor())
        state = encoder.forward(model_pb2.kTrain, inputs)[-2:]
        print state
        deinputs=[labels,state]
        grads = []
        batch_loss = 0
        for output, label in zip(deinputs, labels):
            act = decoder.forward(model_pb2.kTrain, deinputs)
            lvalue = lossfun.forward(model_pb2.kTrain, act, label)
            batch_loss += lvalue.l1()
            grad = lossfun.backward()
            grad /= batch_size
            grad= decoder.backward(model_pb2.kTrain, grad)
            grads.append(grad)
            # print output.l1(), act.l1()
            bar.set_postfix(train_loss=batch_loss / seq_length)
        train_loss += batch_loss

        grads.append(tensor.Tensor())
        grads.append(tensor.Tensor())
        g_rnn_w = decoder.backward(model_pb2.kTrain, grads)[1][0]
        opt.apply_with_lr(epoch, get_lr(epoch), g_rnn_w, decoder_w, 'rnnw')
    print '\nEpoch %d, train loss is %f' % (epoch, train_loss)

