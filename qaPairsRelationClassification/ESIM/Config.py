'''
Created on July 20, 2018
@author : hsiaoyetgun (yqxiao)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import sys


class ModelConfig():
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.arg = None
        self.__addArguments()
        self.__readConfig()

    # add config parameter
    def __addArguments(self):
        # training hyper-parameters
        self.__parser.add_argument('--num_epochs',
                                   '-ep',
                                   default=300,
                                   type=int,
                                   help='Number of epochs')
        self.__parser.add_argument('--batch_size',
                                   '-bs',
                                   default=32,
                                   type=int,
                                   help='Batch size')
        self.__parser.add_argument('--dropout_keep_prob',
                                   '-dkp',
                                   default=1.0,
                                   type=float,
                                   help='Dropout keep probability')
        self.__parser.add_argument('--clip_value',
                                   '-cl',
                                   default=10,
                                   type=float,
                                   help='Norm to clip training')
        self.__parser.add_argument('--learning_rate',
                                   '-lr',
                                   default=0.0004,
                                   type=float,
                                   help='Learning rate')
        self.__parser.add_argument('--l2',
                                   '-l2',
                                   default=0.0,
                                   type=float,
                                   help='L2 normalization constant')
        self.__parser.add_argument('--seq_length',
                                   '-sl',
                                   default=100,
                                   type=int,
                                   help='Max length of data sentence')
        self.__parser.add_argument('--optimizer',
                                   '-op',
                                   default='adam',
                                   choices=['adagrad', 'adadelta', 'adam', 'sgd', 'rmsprop', 'momentum'],
                                   type=str,
                                   help='Optimizer algorithm')
        self.__parser.add_argument('--early_stop_step',
                                   '-ess',
                                   default=50000,
                                   type=int,
                                   help='Early stop condition')

        # embeddings hyper-parameters
        self.__parser.add_argument('--threshold',
                                   '-th',
                                   default=0,
                                   type=int,
                                   help='Cut off freq(word) < threshold in vocabulary')
        self.__parser.add_argument('--embedding_size',
                                   '-es',
                                   default=300,
                                   type=int,
                                   help='Word embedding size')
        self.__parser.add_argument('--embedding_normalize',
                                   '-en',
                                   default=1,
                                   type=int,
                                   help='Normalize word embeddings')

        # layers hyper-parameters
        self.__parser.add_argument('--hidden_size',
                                   '-hs',
                                   default=300,
                                   type=int,
                                   help='Hidden layer size')
        self.__parser.add_argument('--attention_size',
                                   '-as',
                                   default=300,
                                   type=int,
                                   help='Attention layer size')

        # report hyper-parameters
        self.__parser.add_argument('--eval_batch',
                                   '-eb',
                                   default=1000,
                                   type=int,
                                   help='Number of batches between performance reports')

        # IO path
        ## embeddings
        self.__parser.add_argument('--vocab_path',
                                   '-vp',
                                   default='./SNLI/data/vocab.txt',
                                   type=str,
                                   help='Vocabulary file')
        self.__parser.add_argument('--word_path',
                                   '-wp',
                                   default='',
                                   type=str,
                                   help='word file')
        self.__parser.add_argument('--pos_path',
                                   '-pp',
                                   default='',
                                   type=str,
                                   help='pos file')
        self.__parser.add_argument('--embedding_path',
                                   '-embp',
                                   default='',
                                   type=str,
                                   help='Pre-trained word embeddings path')

        ## dataset
        self.__parser.add_argument('--trainset_path',
                                   '-trp',
                                   default='./SNLI/data/train.txt.txt.txt',
                                   type=str,
                                   help='Training set path')
        self.__parser.add_argument('--devset_path',
                                   '-dp',
                                   default='./SNLI/data/dev.txt.txt',
                                   type=str,
                                   help='Validation set path')
        self.__parser.add_argument('--testset_path',
                                   '-tep',
                                   default='./SNLI/data/test.txt',
                                   type=str,
                                   help='Testing set path')

        ## reports
        self.__parser.add_argument('--save_path',
                                   '-sp',
                                   default='./model/checkpoint',
                                   type=str,
                                   help='Directory to save checkpoint')
        self.__parser.add_argument('--best_path',
                                   '-bp',
                                   default='./model/bestval',
                                   type=str,
                                   help='Directory to save the best model')
        self.__parser.add_argument('--best_model_path',
                                   '-bmp',
                                   default='',
                                   type=str,
                                   help='Directory to save the best model')
        self.__parser.add_argument('--log_path',
                                   '-lp',
                                   type=str,
                                   help='Log path')

        ## config
        self.__parser.add_argument('--config_path',
                                   '-cp',
                                   default='./config/config.yaml',
                                   type=str,
                                   help='Config path')

        self.__parser.add_argument('--tfboard_path',
                                   '-tp',
                                   default=' ./tensorboard',
                                   type=str,
                                   help='Config path')

        # prepro
        self.__parser.add_argument('--vocab_size', default=32000, type=int)
        # model
        self.__parser.add_argument('--d_model', default=512, type=int,
                                   help="hidden dimension of encoder/decoder")
        self.__parser.add_argument('--d_ff', default=2048, type=int,
                                   help="hidden dimension of feedforward layer")
        self.__parser.add_argument('--num_blocks', default=6, type=int,
                                   help="number of encoder/decoder blocks")
        self.__parser.add_argument('--num_heads', default=8, type=int,
                                   help="number of attention heads")
        self.__parser.add_argument('--maxlen1', default=100, type=int,
                                   help="maximum length of a source sequence")
        self.__parser.add_argument('--dropout_rate', default=0.3, type=float)
        self.__parser.add_argument('--smoothing', default=0.1, type=float,
                                   help="label smoothing rate")
        self.__parser.add_argument('--warmup_steps', default=4000, type=int)

    # read config information from config file
    def __readConfig(self):
        arg = self.__parser.parse_args()
        with open(arg.config_path) as conf:
            config_dict = yaml.load(conf)
            for key, value in config_dict.items():
                print(key, value)
                sys.argv.append('--' + key)
                sys.argv.append(str(value))
        self.arg = self.__parser.parse_args()

    # print config information
    def print_info(self):
        arg_dict = vars(self.arg)
        print('-' * 20 + ' Config Information ' + '-' * 20)
        for key, value in arg_dict.items():
            print('%-12s : %s' % (key, value))
        print('-' * 60)
