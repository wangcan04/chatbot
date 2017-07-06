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
from word2tensor import load_data,numpy2tensors,labelconvert,answer2vec,question2vec

import time

def get_lr(epoch):
        return 0.001 / float(1 << (epoch / 5))

if __name__ == "__main__":
        # SGD with L2 gradient normalization 
        vocab_size=10000
        word_dim=256
        opt = optimizer.RMSProp(constraint=optimizer.L2Constraint(5))
        cuda = device.create_cuda_gpu_on(1)
        encoder = layer.LSTM(name='lstm1', hidden_size=128, num_stacks=2, dropout=0.5, input_sample_shape=(word_dim,))
        decoder = layer.LSTM(name='lstm2', hidden_size=128, num_stacks=2, dropout=0.5, input_sample_shape=(word_dim,))
        encoder.to_device(cuda)
        decoder.to_device(cuda)
        encoder_w = encoder.param_values()[0]
        encoder_w.uniform(-0.08, 0.08)
        decoder_w = decoder.param_values()[0]
        decoder_w.uniform(-0.08, 0.08)

        dense = layer.Dense('dense', vocab_size, input_sample_shape=(128,))
        dense.to_device(cuda)
        dense_w = dense.param_values()[0]
        dense_b = dense.param_values()[1]
        initializer.uniform(dense_w, dense_w.shape[0], 0)
        dense_b.set_value(0)

        #g_encoder_w = tensor.Tensor(encoder_w.shape, cuda)
        #g_encoder_w.set_value(0.0)
        g_dense_w = tensor.Tensor(dense_w.shape, cuda)
        g_dense_b= tensor.Tensor(dense_b.shape, cuda)
        '''
        g_dense_w.set_value(0.0)
        inlosslistg_dense_b.set_value(0.0)
        '''
        lossfun = loss.SoftmaxCrossEntropy()
        batch_size=50
        maxlength=22
        num_train_batch=5000
        metadata, idx_q, idx_a=load_data()
        num_epoch=100
        batchlosslist=np.zeros([num_epoch,num_train_batch])
        trainlosslist=np.zeros(num_epoch)
        ques = np.load('question_vec.npy')
        ans = np.load('answer_vec.npy')
        for epoch in range(num_epoch):
                train_loss = 0
                bar =range(num_train_batch)
                for b in range (num_train_batch):
                        inputbatch=ques[b * batch_size: (b + 1) * batch_size]
                        print inputbatch.shape
                        inputs= numpy2tensors(inputbatch,dev=cuda)
                        inputs.append(tensor.Tensor())
                        inputs.append(tensor.Tensor())
                        batcha=idx_a[b * batch_size: (b + 1) * batch_size]
                        outputs = encoder.forward(model_pb2.kTrain, inputs)[-2:]
                        #print 'output for origin input:', len(outputs), outputs[0].shape, outputs[1].shape
                        inputs2batch=ans[b * batch_size: (b + 1) * batch_size]
                        print inputs2batch.shape
                        inputs2batch = numpy2tensors(inputs2batch,dev=cuda)
                        inputs2batch = inputs2batch[:-1]
                        inputs2batch.extend(outputs)
                        #print len(inputs2), inputs2[-2].shape, inputs2[-1].shape
                        labels=labelconvert(batcha,cuda)[1:]
                        #print 'gt:', len(labels), labels[0].shape
                        grads=[]
                        batch_loss=0
                        g_dense_w.set_value(0.0)
                        g_dense_b.set_value(0.0)
                        outputs2 = decoder.forward(model_pb2.kTrain, inputs2batch)[0:-2]
                        #print 'output from labeled data:', len(outputs2), outputs2[0].shape
                        for output,label in zip(outputs2,labels):
                                act=dense.forward(model_pb2.kTrain, output)
                                lvalue=lossfun.forward(model_pb2.kTrain,act,label)
                                batch_loss += lvalue.l1()
                                grad=lossfun.backward()
                                grad/=batch_size
                                grad,gwb=dense.backward(model_pb2.kTrain,grad)
                                grads.append(grad)
                                g_dense_w += gwb[0]
                                g_dense_b += gwb[1]
                        batchlosslist[epoch][b]=batch_loss/maxlength
                        train_loss += batch_loss

                        print '\nbatch %d loss is %f' % (b,batch_loss / maxlength)
                        grads.append(tensor.Tensor())
                        grads.append(tensor.Tensor())
                        g_decoder_w = decoder.backward(model_pb2.kTrain, grads)[1][0]
                        g_encoder_w = encoder.backward(model_pb2.kTrain, grads[:-1])[1][0]

                        dense_w, dense_b = dense.param_values()
                        opt.apply_with_lr(epoch, get_lr(epoch), g_decoder_w, decoder_w,'decoderw')
                        #opt.apply_with_lr(epoch, 0.0, g_encoder_w, encoder_w,'encoderw')
                        opt.apply_with_lr(epoch, get_lr(epoch), g_dense_w, dense_w, 'dense_w')
                        opt.apply_with_lr(epoch, get_lr(epoch), g_dense_b, dense_b, 'dense_b')

                        #time.sleep(0.2)
                trainlosslist[epoch]=train_loss /num_train_batch / maxlength
                print '\nEpoch %d, train loss is %f' % (epoch, train_loss /num_train_batch / maxlength)
                with open('%d_trainingmodel128.bin'%(epoch),'wb')as fd:
                    print 'saving model'
                    d={}
                    for name, w in zip(['encoder_w','decoder_w', 'dense_w', 'dense_b'],[encoder_w,decoder_w, dense_w, dense_b]):
                        d[name] = tensor.to_numpy(w)
                    d['hidden_size'] = 128
                    d['num_stacks'] = 2
                    d['dropout'] = 0.5
                    pickle.dump(d, fd)

        np.savetxt("trainresult128.txt", trainlosslist);
