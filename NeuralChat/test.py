import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
import nltk
from six.moves import xrange
import models.chatbot as chatbot
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils
from os import listdir
from os.path import isfile, join
import util.tokenizer
convo_hist_limit = 1
max_source_length = 0
max_target_length = 0
_buckets = [(10, 10), (50, 15), (100, 20), (200, 50)]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay_factor", 0.97, "Learning rate decays by this much.")
flags.DEFINE_float("grad_clip", 5.0, "Clip gradients to this norm.")
flags.DEFINE_float("train_frac", 0.7, "Percentage of data to use for \
        training (rest goes into test set)")
flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
flags.DEFINE_integer("max_epoch", 6, "Maximum number of times to go over training set")
flags.DEFINE_integer("hidden_size", 200, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
flags.DEFINE_integer("vocab_size", 40000, "Max vocabulary size.")
flags.DEFINE_integer("dropout", 0.8, "Probability of hidden inputs being removed between 0 and 1.")
flags.DEFINE_string("data_dir", "data/", "Directory containing processed data.")
flags.DEFINE_string("raw_data_dir", "data/cornell_lines/", "Raw text data directory")
##TODO add more than one tokenizer
flags.DEFINE_string("tokenizer", "basic", "Choice of tokenizer, options are: basic (for now)")
flags.DEFINE_integer("max_train_data_size", 0,
        "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 200,
        "How many training steps to do per checkpoint.")
flags.DEFINE_integer("max_target_length", 50, "max length of target response")
flags.DEFINE_integer("max_source_length", 75, "max length of source input")
flags.DEFINE_integer("convo_limits", 1, "how far back the conversation memory should be")
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/numlayers_2_hsize_200_vsize_40000_max_tlength_50_max_slength_75/', 'Directory to store/restore checkpoints')
FLAGS = tf.app.flags.FLAGS
def main():
        with tf.Session() as sess:
                model = load_model(sess, FLAGS.checkpoint_dir)
                model.batch_size = 1
                model.dropout = 1
                vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
                sys.stdout.write(">")
                sys.stdout.flush()
                sentence = sys.stdin.readline().lower()
                #conversation_history=sentence.decode('utf-8') 
                while sentence:
                        sentence = b'how are you'
                        sentence = util.tokenizer.basic_tokenizer(sentence)
                        #token_ids = list(reversed(vocab.token_2_indices(" ".join(conversation_history))))
                        token_ids = list(reversed(vocab.token_2_indices(sentence)))
                        source = np.zeros(shape=[1, len(token_ids)], dtype=np.int32)

                        for i,j in enumerate(token_ids):
                                source[0,i]=j
                        source_lengths=[]
                        source_lengths.append(1)
                        output_logits = model.test(sess,source,source,source_lengths,source_lengths)

                        #TODO implement beam search
                        #outputs = outputs[:outputs.index(5)]
                        print (output_logits)
                        convo_output=" ".join(vocab.indices_2_tokens(output_logits))
                        conversation_history.append(convo_output)
                        print(convo_output)
                        sys.stdout.write(">")
                        sys.stdout.flush()
                        sentence = sys.stdin.readline().lower()
                        conversation_history.append(sentence)
                        conversation_history = conversation_history[-convo_hist_limit:]
 def load_model(session, path):
        vocab_size=40000
        global max_source_length
        global max_target_length
        params = hyper_params.restore_hyper_params(path)
        max_source_length = params["max_source_length"]
        max_target_length = params["max_target_length"]
        convo_hist_limit = params["convo_limits"]
        model = chatbot.ChatbotModel(vocab_size=vocab_size,
                                                                 hidden_size=FLAGS.hidden_size,
                                                                 dropout=1.0,
                                                                 num_layers=FLAGS.num_layers,
                                                                 max_gradient_norm=FLAGS.grad_clip,
                                                                 batch_size=FLAGS.batch_size,
                                                                 learning_rate=FLAGS.learning_rate,
                                                                 max_target_length = FLAGS.max_target_length,
                                                     max_source_length = FLAGS.max_source_length,
                                                                 lr_decay_factor=FLAGS.lr_decay_factor,
                                                                 decoder_mode=True)
        ckpt_path = tf.train.latest_checkpoint(path)
        if ckpt_path:
                print("Reading model parameters from {0}".format(ckpt_path))
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(session,ckpt_path)
        else:
                print ("No model!")
        return model


if __name__=="__main__":
        main()                       
