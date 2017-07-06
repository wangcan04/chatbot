EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'twitter_en.txt'

limit = {
    'maxq': 20,
    'minq': 0,
    'maxa': 22,
    'mina': 3
}

UNK = 'unk'
VOCAB_SIZE = 9998

import random
import sys

import nltk
import itertools
from collections import defaultdict
import numpy as np

import pickle
import pprint
import gensim
import multiprocessing
def ddefault():
    return 1


'''
 read lines from file
     return [list of lines]
'''


def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


'''
 split sentences in one line
  into multiple lines
    return [list of lines]
'''


def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)
'''


def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )
'''


def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [x[0] for x in vocab]
    # word2index
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )
'''


def filter_data(sequences):
    filtered_q, filtered_a, alllines = [], [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i + 1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= (limit['maxa']-2):
                sequences[i+1]='GO '+ sequences[i+1]+' EOS'
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i + 1])
                alllines.append(sequences[i])
                alllines.append(sequences[i+1])
    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')
    return filtered_q, filtered_a,alllines


'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''


def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        # print(len(idx_q[i]), len(q_indices))
        # print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''


def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0] * (maxlen - len(seq))

def word2vec():
    FILENAME = 'text.txt'
    sentences = gensim.models.word2vec.LineSentence(FILENAME)
    model = gensim.models.Word2Vec(sentences,
                                  size=256,
                                window=30,
                                min_count=1,
                                workers=multiprocessing.cpu_count())
    model.save("word2vec_gensim")
    model.wv.save_word2vec_format("word2vec_org",
                               "vocabulary",
                              binary=False)
    return 'word to vector successful'
def process_data():
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)

    # change to lower case (just for en)
    lines = [line.lower() for line in lines]

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    # filter out unnecessary characters
    print('\n>> Filter lines')
    lines = [filter_line(line, EN_WHITELIST) for line in lines]
    print(lines[121:125])

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines, alllines = filter_data(lines)
    f=open('text.txt','w')
    for i in alllines:
        k = ''.join([str(j) for j in i])
        f.write(k+'\n')
    f.close()
    print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))
    a = word2vec()
    print a
    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [wordlist.split(' ') for wordlist in qlines]
    atokenized = [wordlist.split(' ') for wordlist in alines]
    print('\n:: Sample from segmented list of words')
    print('\nq : {0} ; a : {1}'.format(qtokenized[59], atokenized[59]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)
    print('\nq : {0} ; a : {1}'.format(idx_q[59], idx_a[59]))
    print('\nq : {0} ; a : {1}'.format(idx_q[61], idx_a[61]))

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == '__main__':
    process_data()
    model = gensim.models.Word2Vec.load("word2vec_gensim")
    qlines = open("dataq.txt").read().split('\n')
    qlinesvec=[]
    alinesvec=[]
    max=20
    for i in range(len(qlines)):
        se = qlines[i].split()
        vec = []
        if '' in se:
           se.remove('')
        for x in se:
           vec.append(model.wv[x])
        vec = np.array(vec)
        if len(vec) < max:
            pad = np.zeros((max, 256), dtype=np.float32)
            for i in range(len(vec)):
                 pad[i] = vec[i]
            vec = pad
        qlinesvec.append(vec)
    max=22
    alines = open("dataa.txt").read().split('\n')
    for i in range(len(alines)):
        se = alines[i].split()
        vec = []
        if '' in se:
           se.remove('')
        for x in se:
           vec.append(model.wv[x])
        vec = np.array(vec)
        if len(vec) < max:
            pad = np.zeros((max, 256), dtype=np.float32)
            for i in range(len(vec)):
                 pad[i] = vec[i]
            vec = pad
        alinesvec.append(vec)
    print 'save word vector'
    np.save('question_vec.npy',qlinesvec)
    np.save('answer_vec.npy',alinesvec)
    '''
    file=open('qlines.txt','w')
    file.write('\n'.join(str(num) for num in qlinesvec))
    file.close()
    file=open('alines.txt','w')
    file.write('\n'.join(str(num) for num in alinesvec))
    file.close()
    '''
