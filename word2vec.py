import gensim
import multiprocessing
from singa import tensor
import numpy as np

FILENAME = 'dataclean.txt'


def word2vec():
    sentences = gensim.models.word2vec.LineSentence(FILENAME)
    model = gensim.models.Word2Vec(sentences,
                                  size=100,
                                window=30,
                                min_count=1,
                                workers=multiprocessing.cpu_count())
    model.save("word2vec_gensim")
    model.wv.save_word2vec_format("word2vec_org",
                              "vocabulary",
                              binary=False)

def question2vec(i,max):
    model = gensim.models.Word2Vec.load("word2vec_gensim")
    lines = open("dataclean.txt").read().split('\n')
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

def answer2vec(i):
    model = gensim.models.Word2Vec.load("word2vec_gensim")
    lines = open("dataclean.txt").read().split('\n')
    se = lines[i].split()
    vec = []
    if '' in se:
        se.remove('')
    for x in se:
        vec.append(model.wv[x])
    vec = np.array(vec)
    return vec

def numpy2tensors(npx,npy,dev):
    '''batch, seq, dim -- > seq, batch, dim'''
    tmpx = np.swapaxes(npx, 0, 1)
    tmpy = np.swapaxes(npy, 0, 1)
    inputs = []
    labels = []
    for t in range(tmpx.shape[0]):
        x = tensor.from_numpy(tmpx[t])
        y = tensor.from_numpy(tmpy[t])
        x.to_device(dev)
        y.to_device(dev)
        inputs.append(x)
        labels.append(y)
    return inputs, labels

if __name__ == '__main__':
    word2vec()

