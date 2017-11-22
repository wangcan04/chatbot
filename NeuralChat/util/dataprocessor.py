import os
import util.tokenizer
import util.vocabutils as vocab_utils
from tensorflow.python.platform import gfile
from random import shuffle
from multiprocessing import Process, Lock
import time
from math import floor

class DataProcessor(object):
    def __init__(self, max_vocab_size, source_data_path,
    processed_data_path, train_frac, tokenizer_str,
    num_lines=4, max_target_length=50, max_source_length=200):
        '''
        Inputs:
        max_vocab_size: max size of vocab allowed
        source_data_path: path to raw data files to be processed
        processed_data_path: path to processed data directory (usually just data/)
        train_frac: fraction of data to use for training
        tokenizer_str: string, type of tokenizer to use
        num_lines: max number of lines for conversational history
        max_target_length: max length of target sentence
        max_source_length: max length of source sentence
        '''
        self.MAX_SOURCE_TOKEN_LENGTH = max_source_length
        #subtract 2 for eos and go tokens
        self.MAX_TARGET_TOKEN_LENGTH = max_target_length-2
        self.NUM_LINES = num_lines
        self.tokenizer = util.tokenizer.basic_tokenizer
        assert train_frac > 0.0 and train_frac <= 1.0, "Train frac not between 0 and 1..."
        self.train_frac = train_frac
        self.max_vocab_size = max_vocab_size
        self.source_data_path = source_data_path
        self.processed_data_path = processed_data_path
        train_path = os.path.join(processed_data_path, "train/")
        test_path = os.path.join(processed_data_path, "test/")

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        self.data_source_train = os.path.join(train_path,
            "data_source_train.txt")
        self.data_target_train = os.path.join(train_path,
            "data_target_train.txt")

        self.data_source_test = os.path.join(test_path,
            "data_source_test.txt")
        self.data_target_test = os.path.join(test_path,
            "data_target_test.txt")

        print("Checking to see what data processor needs to do...")
        vocab_path = os.path.join(processed_data_path, "vocab.txt")
        self.vocab_exists = gfile.Exists(vocab_path)

        self.data_files_exist = self.vocab_exists and \
            gfile.Exists(self.data_source_train) and \
            gfile.Exists(self.data_target_train) and \
            gfile.Exists(self.data_source_test) and \
            gfile.Exists(self.data_target_test)

    def run(self):
        if not self.data_files_exist:
            print("Obtaining raw text conversation files...")
            text_files = self.get_raw_file_list(self.source_data_path)
            # randomly shuffle order of files
            shuffle(text_files)
            num_train_files = int(self.train_frac * len(text_files))

        #create vocab file
        if not self.vocab_exists:
            vocab_builder = vocab_utils.VocabBuilder(self.max_vocab_size, self.processed_data_path)
            print("Building vocab...")
            #loop through data
            for text_file in text_files:
                with open(text_file, "rb") as f:
                    vocab_builder.grow_vocab(f.read())
            print("Creating vocab file...")
            vocab_builder.create_vocab_file()

        if not self.data_files_exist:
            self.vocab_mapper = vocab_utils.VocabMapper(self.processed_data_path)
            #create source and target token id files
            processes = []
            print("Creating token id data source and target train files...")

            if len(text_files) == 1:
                num_train_files = 1
                text_files = self.split_single_2_many(text_files[0], self.train_frac)

            p1 = Process(target=self.loop_parse_text_files, args=([text_files[:num_train_files]], True))
            p1.start()
            processes.append(p1)
            print("Creating token id data source and target test files...")
            print("This is going to take a while...")
            p2 = Process(target=self.loop_parse_text_files, args=([text_files[num_train_files:]], False))
            p2.start()
            processes.append(p2)

            for p in processes:
                if p.is_alive():
                    p.join()

            print("Done data pre-processing...")

    def loop_parse_text_files(self, text_files, is_train):
        for text_file in text_files[0]:
            self.parse_text_file(text_file, is_train)

    def split_single_2_many(self, text_file, train_frac):
        '''
        Split a single data file into many files
        (to work into processing pipeline)
        '''
        temp = "temp/"
        if not gfile.Exists(temp):
            os.mkdir(temp)
        with open(text_file, 'r') as f:
            sentences = f.read().split('\n')
            num_train = int(floor(train_frac * len(sentences)))
            if num_train %2 != 0:
                num_train += 1
            num_test = len(sentences) - num_train
            print("num train {0}, num test {1}".format(num_train, num_test))
            train_file_name = "{0}{1}train.txt".format(temp,int(time.time()))
            test_file_name = "{0}{1}test.txt".format(temp,int(time.time()))
            with open(train_file_name, "w+") as f2:
                f2.write("\n".join(sentences[:num_train]))
            with open(test_file_name, "w+") as f2:
                f2.write("\n".join(sentences[num_train:]))
            return [train_file_name, test_file_name]

    def parse_text_file(self, text_file, is_train):
        with open(text_file, "rb") as f:
            line_buffer = []
            for line in f:
                if len(line_buffer) > self.NUM_LINES:
                    self.find_sentence_pairs(line_buffer, is_train)
                    line_buffer.pop(0)
                line_buffer.append(line)

    def get_raw_file_list(self, path):
        text_files = []
        for f in os.listdir(path):
            if not f.endswith("~") or not f == "":
                text_files.append(os.path.join(path, f))
        return text_files


    def find_sentence_pairs(self, line_buffer, is_train):
        assert len(line_buffer) == self.NUM_LINES+1, "Num lines: {0}, length of line buffer: {1}".format(self.NUM_LINES, len(line_buffer))
        if len(line_buffer) > 0:
            for i in range(1, len(line_buffer)):
                source_sentences = b" ".join(line_buffer[:i])
                source_sentences = source_sentences.strip()
                target_sentences = line_buffer[i].strip()
                #Tokenize sentences
                source_sentences = self.tokenizer(source_sentences)
                target_sentences = self.tokenizer(target_sentences)

                #Convert tokens to id string, reverse source inputs
                source_sentences = list(reversed(self.vocab_mapper.token_2_indices(source_sentences)))
                target_sentences = self.vocab_mapper.token_2_indices(target_sentences)
                #remove outliers (really long sentences) from data
                if len(source_sentences) >= self.MAX_SOURCE_TOKEN_LENGTH or \
                    len(target_sentences) >= self.MAX_TARGET_TOKEN_LENGTH:
                    print("skipped {0} and {1}".format(len(source_sentences), len(target_sentences)))
                    continue
                source_sentences = " ".join([str(x) for x in source_sentences])
                target_sentences = " ".join([str(x) for x in target_sentences])

                data_source = self.data_source_train
                data_target = self.data_target_train
                if not is_train:
                    data_source = self.data_source_test
                    data_target = self.data_target_test

                with open(data_source, "a+") as f2:
                    f2.write(source_sentences + "\n")
                with open(data_target, "a+") as f2:
                    f2.write(target_sentences + "\n")
